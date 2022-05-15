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

#include "LayerNorm.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/function/FHeader.h"
#include "Embedding.h"
#include "/home/wusiming/niutensor-debug-tools/XDebug.cuh"

/* the nmt namespace */
namespace nmt {

/* set the training flag */
void LayerNorm::SetTrainingFlag(bool myIsTraining) {
  isTraining = myIsTraining;
}

/* constructor */
LayerNorm::LayerNorm() {
  d = 0;
  devID = -1;
  isTraining = false;
  isL1Normed = false;
}

/* de-constructor */
LayerNorm::~LayerNorm() {}

/*
initialize the model
>> config - configuration of the model
>> myDevID - the device id
>> hiddenSize - the hidden size of layer normalization
>> myL1Normed - whether use L1-Norm
*/
void LayerNorm::InitModel(NMTConfig &config, int myDevID, int hiddenSize,
                          bool myL1Normed) {
  SetTrainingFlag(config.training.isTraining);
  d = hiddenSize;
  devID = myDevID;
  isL1Normed = myL1Normed;

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
XTensor LayerNorm::Run(XTensor &input) {
  if (isL1Normed)
    return RunL1Norm(input);
  else
    return RunL2Norm(input);
}

/*
run standard layernorm with l2-norm
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LayerNorm::RunL2Norm(XTensor &input) {
  XTensor &x = input;
  XTensor xn;
  XTensor mean;
  XTensor variance;
  XTensor standard;
  XTensor meanFilled;
  XTensor standardFilled;

  TENSOR_DATA_TYPE dataType = input.dataType;
  
  Welford(input, mean, variance, x.order - 1);
#if 0
    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);
#endif

  /* convert the input and mean to FP32 to escape overflow */
  if (dataType == X_FLOAT16) {
    x = ConvertDataType(x, X_FLOAT);
    mean = ConvertDataType(mean, X_FLOAT);
  }

#if 0
    /* \sigma = (sum_i (x_i - \mu)^2)/m */
    variance = ReduceVariance(x, x.order - 1, mean, false);
#endif

  /* convert to FP16 if needed */
  if (dataType != x.dataType) {
    x = ConvertDataType(x, dataType);
    mean = ConvertDataType(mean, dataType);
    variance = ConvertDataType(variance, dataType);
  }

  /* call the fused function for faster inference */
  if (!isTraining)
    return Normalize(x, x.order - 1, mean, variance, weight, bias, 0.0F);

  /* TODO: add the backward function for Normalize */

  /* standard = sqrt(variance) */
  standard = Power(variance, 0.5F);

  /* unsqueeze mean and standard deviation to fit them into
     the same shape of x */
  meanFilled = Unsqueeze(mean, x.order - 1, x.GetDim(-1), /*inplace=*/false);
  standardFilled =
      Unsqueeze(standard, x.order - 1, x.GetDim(-1), /*inplace=*/isTraining);

  /* x' = (x - \mu)/standard */
  xn = (x - meanFilled) / standardFilled;

  /* result = x' * w + b   */
  xn = xn * weight;

  xn = Sum(xn, bias, /*inplace=*/isTraining);

  return xn;
}

/*
run layernorm with l1-norm (for faster inference)
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LayerNorm::RunL1Norm(XTensor &input) {
  XTensor &x = input;
  XTensor mean;
  XTensor variance;

#if 0
    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    /* \sigma = (sum_i |(x_i - \mu)|)/m */
    variance = ReduceVariance(x, x.order - 1, mean, true);
#endif
  Welford(input, mean, variance, x.order - 1);

  return L1Normalize(x, x.order - 1, mean, variance, weight, bias);
}

} // namespace nmt