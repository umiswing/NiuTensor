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
 * $Created by: Bei Li (libei_neu@outlook.com) 2020-02-03
 * $Modified by: Chi Hu (huchinlp@gmail.com) 2020-12-10
 */

#include "Embedding.h"
#include "LayerNorm.h"
#include "LayerHistory.h"
#include "../../niutensor/tensor/core/CHeader.h"

/* the nmt namespace */
namespace nmt
{

/* set the training flag */
void LayerHistory::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
LayerHistory::LayerHistory()
{
    d = -1;
    devID = -1;
    count = -1;
    nlayer = -1;
    weights = NULL;
    history = NULL;
    layerNorms = NULL;
    isTraining = false;
}

/* de-constructor */
LayerHistory::~LayerHistory()
{
    delete history;
    delete[] layerNorms;
    delete[] weights;
}

/*
initialize the model
>> config - configurations of the model
>> isEnc - indicates whether it is in the encoder
*/
void LayerHistory::InitModel(NMTConfig& config, bool isEnc)
{
    SetTrainingFlag(config.training.isTraining);
    devID = config.common.devID;
    d = isEnc ? config.model.encEmbDim : config.model.decEmbDim;
    nlayer = isEnc ? config.model.encLayerNum : config.model.decLayerNum;

    /*  the triangle weight matrices for all layers
        layer 0: [1, 0, ..., 0]               
        layer 1: [0.5, 0.5, ..., 0]           
        layer 2: [0.33, 0.33, 0.33, ..., 0]   */
    weights = new XTensor[nlayer + 1];
    for (int i = 0; i < nlayer + 1; i++) {
        InitTensor1D(&(weights[i]), i + 1, X_FLOAT, devID);
        if (isTraining) {
            float* data = new float[i + 1];
            for (int j = 0; j < i + 1; j++) {
                data[j] = 1.0F / float(i + 1);
            }
            weights[i].SetData(data, i + 1);
            delete[] data;
        }
    }

    layerNorms = new LayerNorm[nlayer];

    /* initialize the layer normalization of each layer */
    for (int i = 0; i < nlayer; i++) {
        layerNorms[i].InitModel(config, devID, d,
        isEnc ? config.model.encoderL1Norm : config.model.decoderL1Norm);
    }
}

/*
the Add operation
>> layer - the previous layer output. It might be of size B * L * H
           where B = batch size, L = sequence length,
           and H = vector size of each position
*/
void LayerHistory::Add(XTensor& layer)
{
    /* the embedding is not normed */
    count += 1;
    if (history->count == 0) {
        history->Add(layer);
        return;
    }
    XTensor normed;
    normed = layerNorms[count - 2].Run(layer);
    history->Add(normed);
}

/*
calculate the weighted sum of previous layers
the result for the i-th layer is:
result = sum(layers[0...i] * weight[i][0...i])
shape of the result: B * L * H
*/
XTensor LayerHistory::Pop()
{
    TensorList list;
    for (int i = 0; i < history->count; i++) {
        list.Add(&(history->list[i]));
    }
    XTensor stack;
    stack = Merge(list, 0);

    int dimSize[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < stack.order + 1; i++)
        dimSize[i + 1] = stack.dimSize[i];
    dimSize[0] = int(list.Size());
    dimSize[1] /= dimSize[0];

    if (isTraining) {
        XTensor reshapedStack;
        reshapedStack = Reshape(stack, stack.order + 1, dimSize, /*inplace=*/isTraining);

        XTensor multiplication;
        multiplication = MultiplyDim(reshapedStack, weights[list.Size() - 1], 0);

        XTensor res;
        res = ReduceSum(multiplication, 0);

        /* delete unused data to save memory */
        multiplication.DestroyData();

        return res;
    }
    else {
        stack.Reshape(stack.order + 1, dimSize);
        XTensor multiply;
        multiply = MultiplyDim(stack, weights[list.Size() - 1], 0);
        stack.DestroyData();
        if (multiply.dataType == X_FLOAT16) {
            multiply = ConvertDataType(multiply, X_FLOAT);
        }
        XTensor res;
        res = ReduceSum(multiply, 0);
        multiply.DestroyData();
        if (res.dataType != stack.dataType) {
            res = ConvertDataType(res, stack.dataType);
        }
        return res;
    }
}

/* clear the history */
void LayerHistory::ClearHistory(bool reset)
{
    if (history != NULL)
        delete history;
    if (reset)
        history = new History;
    else
        history = NULL;
    count = 0;
}

/* initialize the history */
History::History()
{
    count = 0;
}

/* append a layer to the history */
void History::Add(XTensor& layer)
{
    if (layer.enableGrad && X_ENABLE_GRAD) {
        list[count] = layer;

        /* delete unused data to save memory */
        layer.DestroyData();
    }
    else {
        list[count] = std::move(layer);
    }

    count++;
}

} /* end of the nmt namespace */