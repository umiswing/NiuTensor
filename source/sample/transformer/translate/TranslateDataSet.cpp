#ifdef WIN32
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#endif
#else
#define DBG_NEW new
#endif
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
#include "../../../tensor/XTensor.h"

using namespace nts;

/* the nmt (NiuTrans.NMT) namespace */
namespace nmt {

/* transfrom a line to a sequence */
Sample* TranslateDataset::LoadSample(string line)
{
    const string delimiter = " ";

    /* load tokens and transform them to ids */
    vector<string> srcTokens = SplitString(line, delimiter,
                                           config->model.maxSrcLen - 1);

    IntList* srcSeq = DBG_NEW IntList(int(srcTokens.size()));
    Sample* sample = DBG_NEW Sample(srcSeq);

    for (const string& token : srcTokens) {
        if (srcVocab.token2id.find(token) == srcVocab.token2id.end())
            srcSeq->Add(srcVocab.unkID);
        else
            srcSeq->Add(srcVocab.token2id.at(token));
    }

    /* the sequence should ends with EOS */
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

    SortBySrcLengthDescending();
    XPRINT1(0, stderr, "[INFO] Loaded %d sentences\n", id);

    return true;
}

/* constructor */
TranslateDataset::TranslateDataset()
{
    ifp = NULL;
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

    /* todo: test with different settings */
    const int REQUIRE_MULTIPLE = 8;
    const int MAX_BATCH_SIZE = 1024;
    while (realBatchSize * maxLen * config->translation.beamSize < config->common.wBatchSize
           && realBatchSize < MAX_BATCH_SIZE) {
        realBatchSize++;
    }
    realBatchSize = MAX(8 * (realBatchSize / 8), realBatchSize % 8);

    /* make sure the batch size is valid */
    realBatchSize = MIN(int(buf->Size()) - bufIdx, realBatchSize);

    CheckNTErrors(maxLen != 0, "Invalid length");

    int* batchValues = DBG_NEW int[realBatchSize * maxLen];
    float* paddingValues = DBG_NEW float[realBatchSize * maxLen];

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
        memcpy(&(batchValues[curSrc]), src->items, sizeof(int) * src->Size());
        curSrc += src->Size();

        while (curSrc < maxLen * (i + 1))
            paddingValues[curSrc++] = 0.0F;
    }

    bufIdx += realBatchSize;

    XTensor* batchEnc = (XTensor*)(inputs->Get(0));
    XTensor* paddingEnc = (XTensor*)(inputs->Get(1));
    InitTensor2D(batchEnc, realBatchSize, maxLen, X_INT);
    InitTensor2D(paddingEnc, realBatchSize, maxLen, X_FLOAT);
    batchEnc->SetData(batchValues, batchEnc->unitNum);
    paddingEnc->SetData(paddingValues, paddingEnc->unitNum);

    delete[] batchValues;
    delete[] paddingValues;

    return true;
}

/*
constructor
>> myConfig - configuration of the NMT system
*/
void TranslateDataset::Init(NMTConfig& myConfig)
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
        ifp = DBG_NEW ifstream(config->translation.inputFN);
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
    if (ifp != NULL) {
        ((ifstream*)(ifp))->close();
        delete ifp;
    }
}

} /* end of the nmt (NiuTrans.NMT) namespace */