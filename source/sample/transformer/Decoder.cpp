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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-10-09
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "Config.h"
#include "Decoder.h"
#include "submodel/LayerNorm.h"
#include "submodel/CommonModules.h"
#include "../../tensor/core/CHeader.h"

/* the nmt (NiuTrans.NMT) namespace */
namespace nmt
{
/* set the training flag */
void AttDecoder::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
AttDecoder::AttDecoder()
{
    selfAtts = NULL;
    ffns = NULL;
    selfAttLayerNorms = NULL;
    ffnLayerNorms = NULL;
    enDeAtts = NULL;
    enDeAttLayerNorms = NULL;
    decoderLayerNorm = NULL;
    selfAttCache = NULL;
    enDeAttCache = NULL;
    history = NULL;
    preLN = true;
    useHistory = false;
    finalNorm = false;
    devID = -1;
    embDim = -1;
    nlayer = -1;
    vSize = -1;
    dropoutP = 0.0F;
    isTraining = false;
}

/* de-constructor */
AttDecoder::~AttDecoder()
{
    delete[] selfAttCache;
    delete[] enDeAttCache;
    delete[] selfAtts;
    delete[] ffns;
    delete[] selfAttLayerNorms;
    delete[] ffnLayerNorms;
    delete[] enDeAtts;
    delete[] enDeAttLayerNorms;
    if (finalNorm)
        delete decoderLayerNorm;
    if (useHistory)
        delete history;
}

/*
initialize the model
>> config - configurations of the model
*/
void AttDecoder::InitModel(NMTConfig& config)
{
    devID = config.common.devID;
    nlayer = config.model.decLayerNum;
    embDim = config.model.decEmbDim;
    vSize = config.model.tgtVocabSize;
    preLN = config.model.decPreLN;
    finalNorm = config.model.decFinalNorm;
    useHistory = config.model.useDecHistory;
    dropoutP = config.model.dropout;

    CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");
    CheckNTErrors(vSize > 1, "set vocabulary size by \"-vsizetgt\"");

    selfAtts = new Attention[nlayer];
    ffns = new FFN[nlayer];
    selfAttLayerNorms = new LayerNorm[nlayer];
    enDeAtts = new Attention[nlayer];
    enDeAttLayerNorms = new LayerNorm[nlayer];
    ffnLayerNorms = new LayerNorm[nlayer];

    selfAttCache = new Cache[nlayer];
    enDeAttCache = new Cache[nlayer];

    if (finalNorm)
        decoderLayerNorm = new LayerNorm;

    if (useHistory)
        history = new LayerHistory;

    /* initialize the stacked layers */
    embedder.InitModel(config, false);
    for (int i = 0; i < nlayer; i++) {
        selfAtts[i].InitModel(config, false, true);
        ffns[i].InitModel(config, false);
        selfAttLayerNorms[i].InitModel(devID, embDim);
        ffnLayerNorms[i].InitModel(devID, embDim);
        enDeAtts[i].InitModel(config, false, false);
        enDeAttLayerNorms[i].InitModel(devID, embDim);
        selfAttCache[i].enable = !isTraining;
        enDeAttCache[i].enable = !isTraining;
    }
    if (finalNorm)
        decoderLayerNorm->InitModel(devID, embDim);
    if (useHistory)
        history->InitModel(config);
}

/*
make the decoding network
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> nstep - the current length of the decoder input
<< return - the output tensor of the decoder
*/
XTensor AttDecoder::Make(XTensor& inputDec, XTensor& outputEnc, XTensor* mask,
                         XTensor* maskEncDec, int nstep)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;

    x = embedder.Make(inputDec, true, nstep);

    /* dropout */
    if (isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP, /*inplace=*/true);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        if (useHistory)
            x = history->Pop();

        XTensor att;
        XTensor ende;
        XTensor ffn;
        XTensor res;
        XTensor selfAttnBefore;
        XTensor selfAttnAfter;
        XTensor endeAttnBefore;
        XTensor endeAttnAfter;
        XTensor ffnBefore;

        /* layer normalization with pre-norm for self-attn */
        selfAttnBefore = LN(x, selfAttLayerNorms[i], preLN, true, false);

        /******************/
        /* self attention */
        att = selfAtts[i].Make(selfAttnBefore, selfAttnBefore, selfAttnBefore, 
                               mask, &selfAttCache[i], SELF_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            att = Dropout(att, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(att, x, /*inplace=*/true);

        /* layer normalization with post-norm for self-attention */
        selfAttnAfter = LN(res, selfAttLayerNorms[i], preLN, false, true);

        /* layer normalization with pre-norm for encoder-decoder attention */
        endeAttnBefore = LN(selfAttnAfter, enDeAttLayerNorms[i], preLN, true, false);

        /* encoder-decoder attention */
        ende = enDeAtts[i].Make(outputEnc, endeAttnBefore, outputEnc, maskEncDec, 
                                &enDeAttCache[i], EN_DE_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            ende = Dropout(ende, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(ende, selfAttnAfter, /*inplace=*/true);

        /* layer normalization with post-norm for encoder-decoder attention */
        endeAttnAfter = LN(res, enDeAttLayerNorms[i], preLN, false, true);

        /* layer normalization with pre-norm for ffn */
        ffnBefore = LN(endeAttnAfter, ffnLayerNorms[i], preLN, true, false);

        /* ffn */
        ffn = ffns[i].Make(ffnBefore);

        /* dropout */
        if (isTraining && dropoutP > 0)
            ffn = Dropout(ffn, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(ffn, endeAttnAfter, /*inplace=*/true);

        /* layer normalization with post-norm for ffn */
        x = LN(res, ffnLayerNorms[i], preLN, false, true);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return decoderLayerNorm->RunFast(x);

    return x;
}

/*
run decoding for inference with pre-norm
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> nstep - the current length of the decoder input
<< return - the output tensor of the decoder
*/
XTensor AttDecoder::RunFastPreNorm(XTensor& inputDec, XTensor& outputEnc, XTensor* maskEncDec, int nstep)
{
    XTensor x;

    x = embedder.Make(inputDec, true, nstep);

    for (int i = 0; i < nlayer; i++) {
        XTensor xn;

        /* layer normalization with pre-norm for self-attn */
        xn = selfAttLayerNorms[i].RunFast(x);

        /******************/
        /* self attention */
        xn = selfAtts[i].Make(xn, xn, xn, NULL, &selfAttCache[i], SELF_ATT);

        /* residual connection */
        SumMe(xn, x);

        /* layer normalization with pre-norm for encoder-decoder attention */
        x = enDeAttLayerNorms[i].RunFast(xn);

        /* encoder-decoder attention */
        x = enDeAtts[i].Make(outputEnc, x, outputEnc, maskEncDec,
                             &enDeAttCache[i], EN_DE_ATT);

        /* residual connection */
        SumMe(x, xn);

        /* layer normalization with pre-norm for ffn */
        xn = ffnLayerNorms[i].RunFast(x);

        /* ffn */
        xn = ffns[i].Make(xn);

        /* residual connection */
        SumMe(x, xn);
    }

    if (finalNorm)
        return decoderLayerNorm->RunFast(x);

    return x;
}

/*
run decoding for inference with post-norm
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> nstep - the current length of the decoder input
<< return - the output tensor of the decoder
*/
XTensor AttDecoder::RunFastPostNorm(XTensor& inputDec, XTensor& outputEnc, XTensor* maskEncDec, int nstep)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;

    x = embedder.Make(inputDec, true, nstep);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {
        XTensor xn;

        if (useHistory)
            x = history->Pop();

        /******************/
        /* self attention */
        xn = selfAtts[i].Make(x, x, x, NULL, &selfAttCache[i], SELF_ATT);

        /* residual connection */
        SumMe(xn, x);

        /* layer normalization with post-norm for self-attn */
        xn = selfAttLayerNorms[i].RunFast(xn);

        /* encoder-decoder attention */
        x = enDeAtts[i].Make(outputEnc, xn, outputEnc, maskEncDec,
                             &enDeAttCache[i], EN_DE_ATT);

        /* residual connection */
        SumMe(x, xn);

        /* layer normalization with pre-norm for ffn */
        xn = enDeAttLayerNorms[i].RunFast(x);

        /* ffn */
        x = ffns[i].Make(xn);

        /* residual connection */
        SumMe(x, xn);

        x = ffnLayerNorms->RunFast(x);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return decoderLayerNorm->RunFast(x);

    return x;
}

}