/* NiuTrans.NMT - an open-source neural machine translation system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/*
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#include <iostream>
#include <algorithm>
#include "TranslateDataSet.h"
#include "../../niutensor/tensor/XTensor.h"

using namespace nts;

/* the nmt namespace */
namespace nmt {

#define pow2Num 4096
unsigned int pow2[pow2Num];

/* transfrom a line to a sequence */
Sample* TranslateDataset::LoadSample(const string& line)
{
    const string delimiter = " ";

    /* split a line to tokens by spaces */
    vector<string> srcTokens = SplitString(line, delimiter,
                                           config->model.maxSrcLen - 1);

    IntList* srcSeq = new IntList(int(srcTokens.size()));
    Sample* sample = new Sample(srcSeq);

    /* transform tokens to ids */
    for (const string& token : srcTokens) {
        if (srcVocab.token2id.find(token) == srcVocab.token2id.end())
            srcSeq->Add(srcVocab.unkID);
        else
            srcSeq->Add(srcVocab.token2id.at(token));
    }

    /* the sequence should end with EOS */
    if(srcSeq->Get(-1) != srcVocab.eosID)
        srcSeq->Add(srcVocab.eosID);
    
    return sample;
}

/*
read data from a file to the buffer
*/
bool TranslateDataset::LoadBatchToBuf()
{
    int id = 0;
    ClearBuf();
    emptyLines.Clear();

    string line;

    while (getline(*ifp, line) && id < config->common.bufSize) {

        /* handle empty lines */
        if (line.size() > 0) {
            Sample* sequence = LoadSample(line);
            sequence->index = id;
            buf->Add(sequence);
        }
        else {
            emptyLines.Add(id);
        }

        id++;
    }

    /* hacky code to solve the issue with fp16 */
    appendEmptyLine = false;
    // if (id > 0 && id % 2 != 0) {
    //     line = "EMPTY";
    //     Sample* sequence = LoadSample(line);
    //     sequence->index = id++;
    //     buf->Add(sequence);
    //     appendEmptyLine = true;
    // }

    /* sort the input sequences by length */
    SortBySrcLengthDescending();

    XPRINT1(0, stderr, "[INFO] loaded %d sentences\n", appendEmptyLine ? id - 1 : id);

    return true;
}

void initPower2() {
  pow2[0] = 1;
  pow2[1] = 2;
  size_t i;
  for (i = 2; i < 4; ++i)
    pow2[i] = 4;
  for (; i < 8; ++i)
    pow2[i] = 8;
  for (; i < 16; ++i)
    pow2[i] = 16;
  for (; i < 32; ++i)
    pow2[i] = 32;
  for (; i < 64; ++i)
    pow2[i] = 64;
  for (; i < 128; ++i)
    pow2[i] = 128;
  for (; i < 256; ++i)
    pow2[i] = 256;
  for (; i < 512; ++i)
    pow2[i] = 512;
  for (; i < 1024; ++i)
    pow2[i] = 1024;
  for (; i < 2048; ++i)
    pow2[i] = 2048;
  for (; i < 4096; ++i)
    pow2[i] = 4096;
}
/* constructor */
TranslateDataset::TranslateDataset()
{
    ifp = NULL;
    appendEmptyLine = false;

    initPower2();
}

bool isShapeFit(const int n, const int m, const int devID) {
#ifdef USE_CUDA
  if (!GDevs.GPUs[devID].isInitialized)
    GDevs.GPUs[devID].Init(devID);
  int bXSize = pow2[n-1];
  int bYSize = GDevs.GPUs[devID].GPUMaxThreadNumPerBlock / bXSize;

  int gXSize = int(ceil(float(n) / bXSize));
  int gYSize = int(ceil(float(m) / bYSize));

  return gXSize <= GDevs.GPUs[devID].GPUMaxGridSize[0] &&
         gYSize <= GDevs.GPUs[devID].GPUMaxGridSize[1];
#endif
  return true;
}

/*
load a batch of sequences from the buffer to the host for translating
>> inputs - a list of input tensors (batchEnc and paddingEnc)
   batchEnc - a tensor to store the batch of input
   paddingEnc - a tensor to store the batch of paddings
>> info - the total length and indices of sequences
*/
bool TranslateDataset::GetBatchSimple(XList* inputs, XList* info)
{
    int realBatchSize = 1;

    /* get the maximum sequence length in a mini-batch */
    Sample* longestsample = (Sample*)(buf->Get(bufIdx));
    int maxLen = int(longestsample->srcSeq->Size());

    /* we choose the max-token strategy to maximize the throughput */
    while (realBatchSize * maxLen * config->translation.beamSize <
               config->common.wBatchSize &&
           realBatchSize < config->common.sBatchSize) {
      if (isShapeFit((realBatchSize + 1) * config->translation.beamSize,
                     config->translation.maxLen * config->model.decEmbDim, config->common.devID))
        realBatchSize++;
      else
        break;
    }

    realBatchSize = MIN(realBatchSize, config->common.sBatchSize);

    /* make sure the batch size is valid */
    realBatchSize = MIN(int(buf->Size()) - bufIdx, realBatchSize);

    CheckNTErrors(maxLen != 0, "Invalid length");

    int* batchValues = new int[realBatchSize * maxLen];
    float* paddingValues = new float[realBatchSize * maxLen];

    for (int i = 0; i < realBatchSize * maxLen; i++) {
        batchValues[i] = srcVocab.padID;
        paddingValues[i] = 1.0F;
    }
    
    int* totalLength = (int*)(info->Get(0));
    IntList* indices = (IntList*)(info->Get(1));
    *totalLength = 0;
    indices->Clear();

    /* right padding */
    int curSrc = 0;
    for (int i = 0; i < realBatchSize; ++i) {
        Sample* sequence = (Sample*)(buf->Get(bufIdx + i));
        IntList* src = sequence->srcSeq;
        indices->Add(sequence->index);
        *totalLength += src->Size();

        curSrc = maxLen * i;
        for (int i = 0; i < src->Size(); i++) {
            batchValues[curSrc + i] = src->Get(i);
        }
        curSrc += src->Size();

        while (curSrc < maxLen * (i + 1))
            paddingValues[curSrc++] = 0.0F;
    }

    bufIdx += realBatchSize;

    /* store the data on the CPU */
    XTensor* batchEnc = (XTensor*)(inputs->Get(0));
    XTensor* paddingEnc = (XTensor*)(inputs->Get(1));
    InitTensor2D(batchEnc, realBatchSize, maxLen, X_INT, -1);
    InitTensor2D(paddingEnc, realBatchSize, maxLen, X_FLOAT, -1);
    batchEnc->SetData(batchValues, batchEnc->unitNum);
    paddingEnc->SetData(paddingValues, paddingEnc->unitNum);

    delete[] batchValues;
    delete[] paddingValues;

    return true;
}

/*
constructor
>> myConfig - configuration of the NMT system
>> notUsed - as it is
*/
void TranslateDataset::Init(NMTConfig& myConfig, bool notUsed)
{
    config = &myConfig;

    /* load the source and target vocabulary */
    srcVocab.Load(config->common.srcVocabFN);

    /* share the source and target vocabulary */
    if (strcmp(config->common.srcVocabFN, config->common.tgtVocabFN) == 0)
        tgtVocab.CopyFrom(srcVocab);
    else
        tgtVocab.Load(config->common.tgtVocabFN);

    srcVocab.SetSpecialID(config->model.sos, config->model.eos,
                          config->model.pad, config->model.unk);
    tgtVocab.SetSpecialID(config->model.sos, config->model.eos,
                          config->model.pad, config->model.unk);

    /* translate the content in a file */
    if (strcmp(config->translation.inputFN, "") != 0) {
        ifp = new ifstream(config->translation.inputFN);
        CheckNTErrors(ifp, "Failed to open the input file");
    }
    /* translate the content in stdin */
    else
        ifp = &cin;

    LoadBatchToBuf();
}

/* this is a place-holder function to avoid errors */
Sample* TranslateDataset::LoadSample()
{
    return nullptr;
}

/* check if the buffer is empty */
bool TranslateDataset::IsEmpty() {
    if (bufIdx < buf->Size())
        return false;
    return true;
}

/* de-constructor */
TranslateDataset::~TranslateDataset()
{
    if (ifp != NULL && strcmp(config->translation.inputFN, "") != 0) {
        ((ifstream*)(ifp))->close();
        delete ifp;
    }
}

} /* end of the nmt namespace */