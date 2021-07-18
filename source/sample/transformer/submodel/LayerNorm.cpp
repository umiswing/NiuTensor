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

#include "Embedding.h"
#include "LayerNorm.h"
#include "../../../tensor/core/CHeader.h"

/* the nmt (NiuTrans.NMT) namespace */
namespace nmt
{

/* set the training flag */
void LayerNorm::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
LayerNorm::LayerNorm()
{
    d = 0;
    devID = -1;
    isTraining = false;
}

/* de-constructor */
LayerNorm::~LayerNorm()
{
}

/*
initialize the model
>> myDevID - the device id
>> hiddenSize - the hidden size of layer normalization
*/
void LayerNorm::InitModel(int myDevID, int hiddenSize)
{
    devID = myDevID;

    d = hiddenSize;

    InitTensor1D(&weight, d, X_FLOAT, devID);
    InitTensor1D(&bias, d, X_FLOAT, devID);
    if (isTraining) {
        bias.SetZeroAll();
        weight.SetDataFixed(1);
    }
}

/*
run layernorm for inference
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LayerNorm::RunFast(XTensor& input)
{
    XTensor& x = input;
    XTensor xn;
    XTensor mean;
    XTensor variance;

    TENSOR_DATA_TYPE dataType = input.dataType;

    if (dataType == X_FLOAT16) {
        x = ConvertDataType(x, X_FLOAT);
    }

    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    /* \sigma = (sum_i (x_i - \mu)^2)/m */
    variance = ReduceVariance(x, x.order - 1, mean);

    if (dataType != x.dataType) {
        x = ConvertDataType(x, dataType);
        mean = ConvertDataType(mean, dataType);
        variance = ConvertDataType(variance, dataType);
    }

    xn = Normalize(x, x.order - 1, mean, variance, weight, bias, 0.0F);

    return xn;
}

} /* end of the nmt (NiuTrans.NMT) namespace */