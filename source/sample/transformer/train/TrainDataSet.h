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
 * Here we define the data manager for NMT training.
 * 
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#ifndef __TRAINDATASET_H__
#define __TRAINDATASET_H__

#include "../Config.h"
#include "../DataSet.h"

using namespace std;
using namespace nts;

/* the nmt (NiuTrans.NMT) namespace */
namespace nmt { 

/* The base class of datasets used in NiuTrans.NMT. */
struct TrainDataSet : public DataSetBase
{
private:

    /* training set size */
    int trainingSize;

    /* the pointer to file stream */
    FILE* fp;

    /* index of the current file pointer */
    int fid;

private:

    /* sort buckets by their keys */
    void SortBuckets();

    /* group data into buckets with similar length */
    void BuildBucket();

    /* calculate the batch size according to the number of tokens */
    int DynamicBatching();

    /* load a pair of sequences from the file  */
    Sample* LoadSample() override;

    /* load the samples into the buffer (a list) */
    bool LoadBatchToBuf() override;

public:

    /* reset the file pointer to the begin */
    void ReSetFilePointer();

    /* start the process */
    bool Start();

    /* end the process */
    bool End();

    /* initialization function */
    void Init(NMTConfig& config) override;

    /* load the samples into tensors from the buffer */
    bool GetBatchSimple(XList* inputs, XList* golds) override;

    /* de-constructor */
    ~TrainDataSet();
};

} /* end of the nmt (NiuTrans.NMT) namespace */

#endif /* __DATASET_H__ */