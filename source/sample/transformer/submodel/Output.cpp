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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "Output.h"
#include "Embedding.h"
#include "../../../tensor/core/CHeader.h"

/* the nmt (NiuTrans.NMT) namespace */
namespace nmt
{

/* set the training flag */
void Output::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
Output::Output()
{
    devID = -1;
    vSize = -1;
    hSize = -1;
    isTraining = false;
    shareDecInputOutputEmb = false;
}

/* de-constructor */
Output::~Output()
{
}

/*
initialize the model
>> config - configurations of the model
*/
void Output::InitModel(NMTConfig& config)
{
    devID = config.common.devID;
    hSize = config.model.decEmbDim;
    vSize = config.model.tgtVocabSize;
    shareDecInputOutputEmb = config.model.shareDecInputOutputEmb;

    if (!shareDecInputOutputEmb) {
        InitTensor2D(&w, hSize, vSize, X_FLOAT, devID);

        DTYPE v = 1.0F / (float)sqrt((float)hSize);
        if (isTraining)
            w.SetDataRandn(0, v);
    }
}

/*
make the network (redefined output tensor)
>> input - input tensor
>> output - output tensor
>> normalized - whether ignore the log-softmax
*/
void Output::Make(XTensor& input, XTensor& output, bool normalized)
{
    XTensor& x = input;

    output = MMul(x, X_NOTRANS, w, X_NOTRANS);

    /* use softmax for training */
    if (isTraining) {
        output = Softmax(output, -1);
        return;
    }

    /* normalize the output for beam search */
    if (normalized) {
        auto dataType = output.dataType;
        if (dataType == X_FLOAT16)
            output = ConvertDataType(output, X_FLOAT);

        output = LogSoftmax(output, -1);

        if (output.dataType != dataType)
            output = ConvertDataType(output, dataType);
    }
}

} /* end of the nmt (NiuTrans.NMT) namespace */