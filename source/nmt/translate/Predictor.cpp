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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-13
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include <iostream>

#include "Predictor.h"
#include "../submodel/NNUtil.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{

/* constructor */
StateBundle::StateBundle()
{
    states = NULL;
    isStart = false;
}

/* de-constructor */
StateBundle::~StateBundle()
{
    if (states != NULL)
        delete[] states;
}

/*
create states
>> num - number of states
*/
void StateBundle::MakeStates(int num)
{
    CheckNTErrors(num > 0, "invalid number");

    if (states != NULL)
        delete[] states;

    states = new State[num];

    for (int i = 0; i < num; i++) {
        states[i].prediction = -1;
        states[i].pid = _PID_EMPTY;
        states[i].isEnd = false;
        states[i].isStart = false;
        states[i].isCompleted = false;
        states[i].modelScore = 0;
        states[i].nstep = 0;
        states[i].last = NULL;
    }

    stateNum = num;
}

/* constructor */
Predictor::Predictor()
{
    startSymbol = 2;
}

/* de-constructor */
Predictor::~Predictor()
{
}

/*
create an initial state
>> model - the  model
>> top - the top-most layer of the network
>> input - input of the network
>> beamSize - beam size
>> state - the state to be initialized
*/
void Predictor::Create(NMTModel* model, XTensor* top, const XTensor* input,
                       int beamSize, StateBundle* state)
{
    int dims[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < input->order - 1; i++)
        dims[i] = input->dimSize[i];
    dims[input->order - 1] = beamSize;

    state->nstep = 0.0F;
    state->stateNum = 0;
}

/*
set start symbol
>> symbol - the symbol (in integer)
*/
void Predictor::SetStartSymbol(int symbol)
{
    startSymbol = symbol;
}

/*
read a state
>> model - the  model that keeps the network created so far
>> state - a set of states. It keeps
1) hypotheses (states)
2) probabilities of hypotheses
3) parts of the network for expanding toward the next state
*/
void Predictor::Read(NMTModel* model, StateBundle* state)
{
    m = model;
    s = state;
}

/*
predict the next state
>> next - next states
>> encoding - encoder output, (B, L, E)
>> inputEnc - input of the encoder, (B, L)
>> paddingEnc - padding of the encoder, (B, L)
>> reorderState - the new order of states
>> needReorder - whether we need to the states
>> nstep - current time step of the target sequence
*/
void Predictor::Predict(StateBundle* next, XTensor& encoding,
                        XTensor& inputEnc, XTensor& paddingEnc, XTensor& reorderState, 
                        bool needReorder, int nstep)
{
    int dims[MAX_TENSOR_DIM_NUM];

    /* word indices of positions up to next state */
    XTensor inputDec;

    /* the input of first step is <SOS> */
    if (nstep == 0) {
        InitTensor2D(&inputDec, encoding.GetDim(0), 1, X_INT, inputEnc.devID);
        inputDec.SetDataFixed(startSymbol);
    }
    else {
        /* only pass one step to the decoder */
        inputDec = GetLastPrediction(s, inputEnc.devID);
    }

    if (needReorder) {
        bool removeFinishedCache = reorderState.GetDim(0) < inputDec.GetDim(0);
        if (removeFinishedCache) {
            inputDec = AutoGather(inputDec, reorderState);
        }
        for (int i = 0; i < m->decoder->nlayer; i++) {
            m->decoder->selfAttCache[i].Reorder(reorderState);
            if (removeFinishedCache) {
                m->decoder->enDeAttCache[i].Reorder(reorderState);
            }
        }
    }

    /* prediction probabilities */
    XTensor& output = next->probPath;
    XTensor decoding;

    for (int i = 0; i < inputDec.order - 1; i++)
        dims[i] = inputDec.dimSize[i];
    dims[inputDec.order - 1] = inputDec.dimSize[inputDec.order - 1];

    XTensor maskEncDec;

    /* decoder mask */
    maskEncDec = m->MakeMTMaskDecInference(paddingEnc);

    /* make the decoding network */
    if (m->config->model.decPreLN)
        decoding = m->decoder->RunFastPreNorm(inputDec, encoding, &maskEncDec, nstep);
    else
        decoding = m->decoder->RunFastPostNorm(inputDec, encoding, &maskEncDec, nstep);

    CheckNTErrors(decoding.order >= 2, "The tensor must be of order 2 or larger!");

    /* generate the output probabilities */
    output = m->outputLayer->Make(decoding, true);
}

/*
get the predictions of the previous step
>> state - state bundle of the current step
>> devID - the device id for the predictions
*/
XTensor Predictor::GetLastPrediction(StateBundle* state, int devID)
{
    /* ToDo (huchi): reuse the data instead of reshaping it */
    int dims[] = { state->prediction.unitNum, 1 };
    return Reshape(state->prediction, 2, dims, /*inplace=*/false);
}

} /* end of the nmt namespace */