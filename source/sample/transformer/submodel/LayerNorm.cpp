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
    isL1Normed = false;
}

/* de-constructor */
LayerNorm::~LayerNorm()
{
}

/*
initialize the model
>> myDevID - the device id
>> hiddenSize - the hidden size of layer normalization
>> myL1Normed - whether use L1-Norm
*/
void LayerNorm::InitModel(int myDevID, int hiddenSize, bool myL1Normed)
{
    devID = myDevID;

    isL1Normed = myL1Normed;
    d = hiddenSize;

    InitTensor1D(&weight, d, X_FLOAT, devID);
    InitTensor1D(&bias, d, X_FLOAT, devID);
    if (isTraining) {
        bias.SetZeroAll();
        weight.SetDataFixed(1);
    }
}

/*
initialize the model
>> myDevID - the device id
>> hiddenSize - the hidden size of layer normalization
>> myL1Normed - whether use L1-Norm
*/
XTensor LayerNorm::RunFast(XTensor& input)
{
    if (isL1Normed)
        return RunL1Norm(input);
    else
        return RunL2Norm(input);
}



/*
run layernorm for inference
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LayerNorm::RunL2Norm(XTensor& input)
{
    XTensor& x = input;
    XTensor xn;
    XTensor mean;
    XTensor variance;

    TENSOR_DATA_TYPE dataType = input.dataType;

    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    if (dataType == X_FLOAT16) {
        x = ConvertDataType(x, X_FLOAT);
        mean = ConvertDataType(mean, X_FLOAT);
    }

    /* \sigma = (sum_i (x_i - \mu)^2)/m */
    variance = ReduceVariance(x, x.order - 1, mean, false);

    if (dataType != x.dataType) {
        x = ConvertDataType(x, dataType);
        mean = ConvertDataType(mean, dataType);
        variance = ConvertDataType(variance, dataType);
    }

    xn = Normalize(x, x.order - 1, mean, variance, weight, bias, 0.0F);

    return xn;
}

/*
run layernorm-l1 for inference
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LayerNorm::RunL1Norm(XTensor& input)
{
    XTensor& x = input;
    XTensor mean;
    XTensor variance;

    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    /* \sigma = (sum_i |(x_i - \mu)|)/m */
    variance = ReduceVariance(x, x.order - 1, mean, true);

    return L1Normalize(x, x.order - 1, mean, variance, weight, bias);
}

} /* end of the nmt (NiuTrans.NMT) namespace */