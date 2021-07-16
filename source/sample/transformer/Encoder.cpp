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

#include "Encoder.h"
#include "submodel/LayerNorm.h"
#include "submodel/CommonModules.h"
#include "../../tensor/core/CHeader.h"

/* the nmt (NiuTrans.NMT) namespace */
namespace nmt
{
/* set the training flag */
void AttEncoder::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
AttEncoder::AttEncoder()
{
    devID = -1;
    selfAtts = NULL;
    ffns = NULL;
    attLayerNorms = NULL;
    fnnLayerNorms = NULL;
    encoderLayerNorm = NULL;
    useHistory = false;
    history = NULL;
    dropoutP = 0.0;
    embDim = -1;
    finalNorm = false;
    ignored = -1;
    nlayer = -1;
    preLN = false;
    vSize = -1;
    isTraining = false;
}

/* de-constructor */
AttEncoder::~AttEncoder()
{
    delete[] selfAtts;
    delete[] ffns;
    delete[] attLayerNorms;
    delete[] fnnLayerNorms;
    if (finalNorm)
        delete encoderLayerNorm;
    if (useHistory)
        delete history;
}

/*
initialize the model
>> config - configurations for the model
*/
void AttEncoder::InitModel(NMTConfig& config)
{
    devID = config.common.devID;
    nlayer = config.model.encLayerNum;
    embDim = config.model.encEmbDim;
    vSize = config.model.srcVocabSize;
    preLN = config.model.encPreLN;
    finalNorm = config.model.encFinalNorm;
    useHistory = config.model.useEncHistory;
    dropoutP = config.model.dropout;

    CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");
    CheckNTErrors(vSize > 1, "Set vocabulary size by \"-vsize\"");

    ffns = new FFN[nlayer];
    selfAtts = new Attention[nlayer];
    attLayerNorms = new LayerNorm[nlayer];
    fnnLayerNorms = new LayerNorm[nlayer];

    if (finalNorm)
        encoderLayerNorm = new LayerNorm;

    if (useHistory)
        history = new LayerHistory;

    /* initialize the stacked layers */
    embedder.InitModel(config);
    for (int i = 0; i < nlayer; i++) {
        selfAtts[i].InitModel(config, true, true);
        ffns[i].InitModel(config, true);
        attLayerNorms[i].InitModel(devID, embDim);
        fnnLayerNorms[i].InitModel(devID, embDim);
    }
    if (finalNorm)
        encoderLayerNorm->InitModel(devID, embDim);
    if (useHistory)
        history->InitModel(config);
}

/*
make the encoding network
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
>> maskEncDec - no use
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::Make(XTensor& input, XTensor* mask, XTensor& maskEncDec)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;
    x = embedder.Make(input, false, 0);

    /* dropout */
    if (isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP, /*inplace=*/true);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        if (useHistory)
            x = history->Pop();

        XTensor att;
        XTensor fnn;
        XTensor res;
        XTensor attnBefore;
        XTensor attnAfter;
        XTensor fnnBefore;

        /* layer normalization with pre-norm for self-attn */
        attnBefore = LN(x, attLayerNorms[i], preLN, true, false);

        /* self attention */
        att = selfAtts[i].Make(attnBefore, attnBefore, attnBefore, mask, NULL, SELF_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            att = Dropout(att, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(att, x, /*inplace=*/true);

        /* layer normalization with post-norm for self-attn */
        attnAfter = LN(res, attLayerNorms[i], preLN, false, true);

        /* layer normalization with pre-norm for fnn */
        fnnBefore = LN(attnAfter, fnnLayerNorms[i], preLN, true, false);

        /* fnn */
        fnn = ffns[i].Make(fnnBefore);

        /* dropout */
        if (isTraining && dropoutP > 0)
            fnn = Dropout(fnn, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(fnn, attnAfter, /*inplace=*/true);

        /* layer normalization with post-norm for fnn */
        x = LN(res, fnnLayerNorms[i], preLN, false, true);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return encoderLayerNorm->RunFast(x);

    return x;
}

/*
make the encoding network (wrapper)
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::Make(XTensor& input, XTensor* mask)
{
    XTensor nothing;

    return Make(input, mask, nothing);
}

/* 
run encoding for inference with pre-norm
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::RunFastPreNorm(XTensor& input, XTensor* mask)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;
    x = embedder.Make(input, false, 0);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        if (useHistory)
            x = history->Pop();

        XTensor selfAtt;
        XTensor ffn;
        XTensor selfAttBefore;
        XTensor ffnBefore;

        /* layer normalization with pre-norm for self-attn */
        selfAttBefore = attLayerNorms[i].RunFast(x);

        /* self attention */
        selfAtt = selfAtts[i].Make(selfAttBefore, selfAttBefore, selfAttBefore, mask, NULL, SELF_ATT);

        /* residual connection */
        SumMe(selfAtt, x);

        /* layer normalization with pre-norm for ffn */
        ffnBefore = fnnLayerNorms[i].RunFast(selfAtt);

        /* ffn */
        ffn = ffns[i].Make(ffnBefore);

        /* residual connection */
        x = ffn + selfAtt;

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return encoderLayerNorm->RunFast(x);

    return x;
}

/*
run encoding for inference with post-norm
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::RunFastPostNorm(XTensor& input, XTensor* mask)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;
    x = embedder.Make(input, false, 0);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        if (useHistory)
            x = history->Pop();

        XTensor selfAtt;

        /* self attention */
        selfAtt = selfAtts[i].Make(x, x, x, mask, NULL, SELF_ATT);

        /* residual connection */
        SumMe(selfAtt, x);

        /* layer normalization with post-norm for self-attn */
        selfAtt = attLayerNorms[i].RunFast(selfAtt);

        /* ffn */
        x = ffns[i].Make(selfAtt);

        /* residual connection */
        SumMe(x, selfAtt);

        /* layer normalization with post-norm for ffn */
        x = fnnLayerNorms[i].RunFast(x);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return encoderLayerNorm->RunFast(x);

    return x;
}

} /* end of the nmt (NiuTrans.NMT) namespace */