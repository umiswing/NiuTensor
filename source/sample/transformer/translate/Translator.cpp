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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-27
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#include <iostream>
#include <algorithm>
#include "Searcher.h"
#include "Translator.h"
#include "../../../tensor/XTensor.h"
#include "../../../tensor/XUtility.h"
#include "../../../tensor/core/CHeader.h"

using namespace nts;

/* the nmt (NiuTrans.NMT) namespace */
namespace nmt
{

/* constructor */
Translator::Translator()
{
    config = NULL;
    model = NULL;
    seacher = NULL;
    outputBuf = DBG_NEW XList;
}

/* de-constructor */
Translator::~Translator()
{
    if (config->translation.beamSize > 1)
        delete (BeamSearch*)seacher;
    else
        delete (GreedySearch*)seacher;
    delete outputBuf;
}

/* initialize the model */
void Translator::Init(NMTConfig& myConfig, NMTModel& myModel)
{
    model = &myModel;
    config = &myConfig;

    if (config->translation.beamSize > 1) {
        LOG("Translating with beam search (beam=%d, batchSize=%d, lenAlpha=%.2f, maxLenAlpha=%.2f) ", 
            config->translation.beamSize, config->common.sBatchSize,
            config->translation.lenAlpha, config->translation.maxLenAlpha);
        seacher = DBG_NEW BeamSearch();
        ((BeamSearch*)seacher)->Init(myConfig);
    }
    else if (config->translation.beamSize == 1) {
        LOG("Translating with greedy search (batchSize=%d, maxLenAlpha=%.2f)", 
            config->common.sBatchSize, config->translation.maxLenAlpha);
        seacher = DBG_NEW GreedySearch();
        ((GreedySearch*)seacher)->Init(myConfig);
    }
    else {
        CheckNTErrors(false, "Invalid beam size\n");
    }
}

/* sort the outputs by the indices (in ascending order) */
void Translator::SortOutputs()
{
    sort(outputBuf->items, outputBuf->items + outputBuf->count,
        [](void* a, void* b) {
            return ((Sample*)(a))->index <
                   ((Sample*)(b))->index;
        });
}

/* 
translate a batch of sequences 
>> batchEnc - the batch of inputs
>> paddingEnc - the paddings of inputs
>> indices - indices of input sequences
the results will be saved in the output buffer
*/
void Translator::TranslateBatch(XTensor& batchEnc, XTensor& paddingEnc, IntList& indices)
{
    int batchSize = batchEnc.GetDim(0);
    for (int i = 0; i < model->decoder->nlayer; ++i) {
        model->decoder->selfAttCache[i].miss = true;
        model->decoder->enDeAttCache[i].miss = true;
    }

    IntList** outputs = DBG_NEW IntList * [batchSize];
    for (int i = 0; i < batchSize; i++)
        outputs[i] = DBG_NEW IntList();

    /* greedy search */
    if (config->translation.beamSize == 1) {
        ((GreedySearch*)seacher)->Search(model, batchEnc, paddingEnc, outputs);
    }

    /* beam search */
    if (config->translation.beamSize > 1) {
        XTensor score;
        ((BeamSearch*)seacher)->Search(model, batchEnc, paddingEnc, outputs, score);
    }

    /* save the outputs to the buffer */
    for (int i = 0; i < batchSize; i++) {
        Sample* sample = DBG_NEW Sample(NULL, outputs[i]);
        sample->index = indices[i];
        outputBuf->Add(sample);
    }

    delete[] outputs;
}

/* the translation function */
bool Translator::Translate()
{
    batchLoader.Init(*config);

    /* inputs */
    XTensor batchEnc;
    XTensor paddingEnc;

    /* sentence information */
    XList info;
    XList inputs;
    int wordCount;
    IntList indices;
    inputs.Add(&batchEnc);
    inputs.Add(&paddingEnc);
    info.Add(&wordCount);
    info.Add(&indices);

    while (!batchLoader.IsEmpty()) {
        fprintf(stderr, "%d/%d\n", batchLoader.bufIdx, batchLoader.buf->Size());
        batchLoader.GetBatchSimple(&inputs, &info);
        batchEnc.FlushToDevice(config->common.devID);
        paddingEnc.FlushToDevice(config->common.devID);
        TranslateBatch(batchEnc, paddingEnc, indices);
    }

    /* handle empty lines */
    for (int i = 0; i < batchLoader.emptyLines.Size(); i++) {
        Sample* sample = DBG_NEW Sample(NULL, NULL);
        sample->index = batchLoader.emptyLines[i];
        outputBuf->Add(sample);
    }
    SortOutputs();

    if (strcmp(config->translation.outputFN, "") != 0)
        DumpResToFile(config->translation.outputFN);
    else
        DumpResToStdout();

    for (int i = 0; i < outputBuf->Size(); i++) {
        Sample* s = (Sample*)(outputBuf->GetItem(i));
        delete s;
    }
    outputBuf->Clear();

    return true;
}

/* dump the translation results to a file */
void Translator::DumpResToFile(const char* ofn)
{
    string buffer;
    for (int i = 0; i < outputBuf->Size(); i++) {
        Sample* sample = (Sample*)outputBuf->Get(i);
        if (sample->tgtSeq != NULL) {
            for (int j = 0; j < sample->tgtSeq->Size(); j++) {
                int id = sample->tgtSeq->Get(j);
                buffer += batchLoader.tgtVocab.id2token[id];
                buffer += " ";
            }
        }
        buffer += "\n";
    }
    FILE* f = fopen(ofn, "w");
    fwrite(buffer.c_str(), 1, buffer.length() - 1, f);
}

/* dump the translation results to stdout */
void Translator::DumpResToStdout()
{
    string buffer;
    for (int i = 0; i < outputBuf->Size(); i++) {
        Sample* sample = (Sample*)outputBuf->Get(i);
        if (sample->tgtSeq != NULL) {
            for (int j = 0; j < sample->tgtSeq->Size(); j++) {
                int id = sample->tgtSeq->Get(j);
                buffer += batchLoader.tgtVocab.id2token[id];
                buffer += " ";
            }
        }
        buffer += "\n";
    }
    fwrite(buffer.c_str(), 1, buffer.length() - 1, stdout);
}

} /* end of the nmt (NiuTrans.NMT) namespace */