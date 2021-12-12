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

#include "Searcher.h"
#include "../Config.h"
#include "../../niutensor/tensor/core/CHeader.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{
/* constructor */
BeamSearch::BeamSearch()
{
    alpha = 0;
    maxLen = 0;
    beamSize = 0;
    batchSize = 0;
    endSymbolNum = 0;
    fullHypos = NULL;
    endSymbols = new int[32];
    startSymbol = -1;
    isEarlyStop = false;
    needReorder = false;
    scalarMaxLength = 0.0F;
}

/* de-constructor */
BeamSearch::~BeamSearch()
{
    if (fullHypos != NULL)
        delete[] fullHypos;
    if (endSymbols != NULL)
        delete[] endSymbols;
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void BeamSearch::Init(NMTConfig& config)
{
    maxLen = config.translation.maxLen;
    beamSize = config.translation.beamSize;
    batchSize = config.common.sBatchSize;
    alpha = config.translation.lenAlpha;
    endSymbols[0] = config.model.eos;
    startSymbol = config.model.sos;
    scalarMaxLength = config.translation.maxLenAlpha;

    if (endSymbols[0] >= 0)
        endSymbolNum = 1;
}

/*
prepare for search
>> batchSize - size of the batch
>> beamSize - size of the beam
*/
void BeamSearch::Prepare(int myBatchSize, int myBeamSize)
{
    batchSize = myBatchSize;
    beamSize = myBeamSize;
    needReorder = false;

    /* prepare for the heap of hypotheses */
    if (fullHypos != NULL)
        delete[] fullHypos;

    fullHypos = new XHeap<MIN_HEAP, float>[batchSize];

    for (int i = 0; i < batchSize; i++)
        fullHypos[i].Init(beamSize);
}

/*
search for the most promising states
>> model - the transformer model
>> input - input of the model
>> padding - padding of the input
>> outputs - outputs that represent the sequences as rows
>> score - score of the sequences
*/
void BeamSearch::Search(NMTModel* model, XTensor& input, XTensor& padding, 
                        IntList** outputs, XTensor& score)
{
    Predictor predictor;
    XTensor maskEnc;
    XTensor encoding;
    XTensor encodingBeam;
    XTensor inputBeam;
    XTensor paddingBeam;

    CheckNTErrors(endSymbolNum > 0, "The search class is not initialized!");
    CheckNTErrors(startSymbol >= 0, "The search class is not initialized!");

    Prepare(input.GetDim(0), beamSize);

    input.SetDevice(model->devID);
    padding.SetDevice(model->devID);

    /* encoder mask */
    model->MakeMTMaskEnc(padding, maskEnc);

    /* make the encoding network */
    if (model->config->model.encPreLN)
        encoding = model->encoder->RunFastPreNorm(input, &maskEnc);
    else
        encoding = model->encoder->RunFastPostNorm(input, &maskEnc);

    encodingBeam = Unsqueeze(encoding, encoding.order - 2, beamSize);
    inputBeam = Unsqueeze(input, input.order - 1, beamSize);
    paddingBeam = Unsqueeze(padding, padding.order - 1, beamSize);

    encodingBeam.ReshapeMerged(encodingBeam.order - 4);
    inputBeam.ReshapeMerged(inputBeam.order - 3);
    paddingBeam.ReshapeMerged(paddingBeam.order - 3);

    /* max output-length = scalar * source-length */
    int lengthLimit = int(float(input.GetDim(-1)) * scalarMaxLength) + maxLen;

    CheckNTErrors(lengthLimit > 0, "no max length specified!");

    StateBundle* states = new StateBundle[lengthLimit + 1];
    StateBundle* first = states;
    StateBundle* cur = NULL;
    StateBundle* next = NULL;

    /* create the first state */
    predictor.Create(model, &encodingBeam, &input, beamSize, first);
    predictor.SetStartSymbol(startSymbol);

    first->isStart = true;

    XTensor aliveStates;
    XTensor reorderState;
    InitTensor1D(&reorderState, batchSize * beamSize, X_INT, input.devID);

    /* generate the sequence from left to right */
    for (int l = 0; l < lengthLimit; l++) {

        cur = states + l;
        next = states + l + 1;

        /* read the current state */
        predictor.Read(model, cur);

        /* predict the next state */
        predictor.Predict(next, encodingBeam, inputBeam,
                          paddingBeam, reorderState, needReorder, l);

        /* compute the model score (given the prediction probability) */
        Score(cur, next);

        /* beam pruning */
        Generate(cur, next);

        /* expand the search graph */
        Expand(cur, next, reorderState);

        /* push complete hypotheses into the heap */
        Collect(next);

        /* stop searching when all hypotheses are completed */
        if (IsAllCompleted(next)) {
            l = lengthLimit;
            break;
        }

        XTensor aliveStates;
        aliveStates = GetAliveStates(next);

        if (needReorder) {

            /* remove finished states */
            if (aliveStates.unitNum > 0 && aliveStates.unitNum < 0.5 * reorderState.unitNum) {
                /*reorderState = AutoGather(reorderState, aliveStates);
                next->probPath.Reshape(next->probPath.unitNum);
                next->probPath = AutoGather(next->probPath, aliveStates);*/
            }
            inputBeam = AutoGather(inputBeam, reorderState);
            paddingBeam = AutoGather(paddingBeam, reorderState);
            encodingBeam = AutoGather(encodingBeam, reorderState);
            
        }
    }

    /* fill the heap with incomplete hypotheses if necessary */
    FillHeap(next);

    Dump(outputs, &score);

    delete[] states;
}

/*
compute the model score for each hypotheses
>> prev - the beam of the previous state
>> beam - the beam that keeps a number of states
*/
void BeamSearch::Score(StateBundle* prev, StateBundle* beam)
{
    XTensor& score = beam->modelScore;
    XTensor& probPath = beam->probPath;
    XTensor& probPathPrev = prev->probPath;

    int order = probPath.order;
    int outputSize = probPath.dimSize[probPath.order - 1];
    int dims[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < order; i++)
        dims[i] = probPath.dimSize[i];

    if (probPath.dataType == X_FLOAT16)
        probPath = ConvertDataType(probPath, X_FLOAT);

    if (prev->isStart) {
        InitTensor1D(&probPathPrev, probPath.GetDim(0), probPath.dataType, probPath.devID);
        probPathPrev.SetZeroAll();
    }

    probPathPrev.Reshape(probPathPrev.unitNum);
    probPath.Reshape(probPath.unitNum / outputSize, outputSize);

    /* the log-scale probability of the entire sequence */
    SumDim(probPath, probPathPrev, score, 0);

    beam->nstep = prev->nstep + 1.0F;

    if (prev->isStart) {
        XTensor firstMask = MakeFirstMask(beam);
        firstMask.Reshape(firstMask.unitNum);

        /* mask the hypotheses in the beam except the first one */
        SumDim(score, firstMask, score, 0);
    }

    score.Reshape(order, dims);
}

/*
generate tokens for the next state via beam pruning
>> prev - the last beam
>> beam - the beam that keeps a number of states
*/
void BeamSearch::Generate(StateBundle* prev, StateBundle* beam)
{
    int dimsBeam[MAX_TENSOR_DIM_NUM];
    int dimsTopK[MAX_TENSOR_DIM_NUM];

    XTensor& score = beam->modelScore;
    XTensor& index = beam->prediction;
    XTensor& preID = beam->preID;
    XTensor& probPath = beam->probPath;

    int order = score.order;

    for (int i = 0; i < order; i++) {
        dimsBeam[i] = score.dimSize[i];
        dimsTopK[i] = score.dimSize[i];
    }

    int sizeVocab = score.dimSize[score.order - 1];

    dimsBeam[order - 3] /= beamSize;
    dimsBeam[order - 1] *= beamSize;
    dimsTopK[order - 3] = dimsBeam[order - 3];
    dimsTopK[order - 1] = beamSize;

    InitTensor(&probPath, order, dimsTopK, score.dataType, score.devID);
    InitTensor(&index, order, dimsTopK, X_INT, score.devID);

    score.Reshape(order, dimsBeam);

    /* keep the most promising candidates in the beam */
    TopK(score, probPath, index, -1, beamSize, true);

    /* "preID" represents the id (or the offset) of the previous state used to make the current
       hypotheses. Note that we reshape the "score" tensor into a matrix where each
       row means a previous state. The column number is size-of-beam \times vocab-size. We,
       therefore, divide entries of the top-k index by vocab-size to compute the id of the
       previous state for each hypotheses in the top-k list. */
    preID = Descale(index, sizeVocab);

    /* Then, we do something similar to "preID". For the top-k predictions, we need
       to know their indices in the vocabulary. We compute the offset of each prediction
       in the vocabulary by dividing it with vocab-size and computing the remainder. */
    ModMe(index, sizeVocab);

    /* the GNMT-like length penalty */
    float lp = LengthPenalizer::GNMT(beam->nstep, alpha);
    score = probPath / lp;
}

/*
expand the search graph
>> prev - the last beam
>> beam - the beam that keeps a number of states
>> reorderState - the new order of states
*/
void BeamSearch::Expand(StateBundle* prev, StateBundle* beam, XTensor& reorderState)
{
    beam->MakeStates(beam->prediction.unitNum);

    State* states = beam->states;
    XTensor& idRef = beam->preID;
    XTensor& modelScoreRef = beam->modelScore;
    XTensor& predictionRef = beam->prediction;
    XTensor id;
    XTensor modelScore;
    XTensor prediction;
    XTensor reorderStateCPU;

    InitTensorOnCPU(&id, &idRef);
    InitTensorOnCPU(&prediction, &predictionRef);
    InitTensorOnCPU(&reorderStateCPU, &reorderState);

    if (beam->probPath.dataType == X_FLOAT) {
        InitTensorOnCPU(&modelScore, &modelScoreRef);
        CopyValues(modelScoreRef, modelScore);
    }
    else {
        modelScore = ConvertDataType(modelScoreRef, X_FLOAT);
        modelScore.SetDevice(-1);
    }

    /* we copy the data to CPU because the frequent access to GPU is slow
       and we can speed up the process by doing the job on CPU. */
    CopyValues(idRef, id);
    CopyValues(predictionRef, prediction);

    CheckNTErrors(beam->stateNum == id.unitNum, "Errors occur in counting!");

    /* Related variables are kept on the states of the graph. All these are
       maintained on CPUs to ease the implementation of frequent access and
       modification of the states. An alternative is to do this on GPUs but
       it needs much more coding work and the speed-up is not obvious. */
    bool reorder = false;
    for (int i = 0; i < beam->stateNum; i += beamSize) {
        
        for (int j = 0; j < beamSize; j++) {
            int k = i + j;
            State& state = states[k];

            int offset = id.GetInt(k);
            int pid = i / beamSize;
            reorderStateCPU.SetInt(i + offset, k);
            if (offset != j)
                reorder = true;

            State* last = prev->states + pid * beamSize + offset;

            /* pointer to the previous state */
            if (prev->isStart) {
                state.last = NULL;
                state.pid = pid;
                state.nstep = 0;
                state.isCompleted = false;
            }
            else {
                state.last = last;
                state.pid = state.last->pid;
                state.nstep = last->nstep + 1;
                state.isCompleted = last->isCompleted;
            }

            /* scores */
            state.modelScore = modelScore.Get(k);

            /* prediction */
            state.prediction = prediction.GetInt(k);

            /* check if it is the end of the sequence */
            state.isEnd = IsEnd(state.prediction);
            state.isCompleted = (state.isCompleted || state.isEnd);
        }
    }

    //for (int i = 0; i < beam->stateNum; i += beamSize) {
    //    int bestID = -1;
    //    float bestScore = -9999.0F;
    //    for (int j = 0; j < beamSize; j++) {
    //        int k = i + j;
    //        State& state = states[k];

    //        /* find the beam that is finished and has the highest score */
    //        if (state.modelScore > bestScore) {
    //            if(state.isCompleted)
    //                bestID = j;
    //            bestScore = state.modelScore;
    //        }
    //    }

    //    /* force other beams that have lower scores to be finished */
    //    if (bestID != -1) {
    //        for (int j = 0; j < beamSize; j++) {
    //            int k = i + j;
    //            State& state = states[k];
    //            state.isCompleted = true;
    //        }
    //    }
    //}
    
    needReorder = reorder;
    if(needReorder)
        CopyValues(reorderStateCPU, reorderState);
}

/*
collect hypotheses with ending symbols. Given a beam of hypotheses,
we remove the finished hypotheses and keep them in a heap.
>> beam  - the beam that keeps a number of states
*/
void BeamSearch::Collect(StateBundle* beam)
{
    State* states = beam->states;

    for (int i = 0; i < beam->stateNum; i++) {
        State& state = states[i];

        CheckNTErrors(state.pid >= 0 && state.pid < batchSize,
            "Invalid sample id!");

        /* check if this is the first end symbol. It is false
           if there have been end symbols in previously generated words. */
        bool isCompleted = state.isCompleted && 
             (state.last == NULL || !state.last->isCompleted);

        /* we push the hypothesis into the heap when it is completed */
        if ((state.isEnd || state.isCompleted)) {
            fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
        }
    }
}

/*
fill the hypothesis heap with incomplete hypotheses
>> beam  - the beam that keeps a number of states (final)
*/
void BeamSearch::FillHeap(StateBundle* beam)
{
    State* states = beam->states;

    for (int i = 0; i < beam->stateNum / beamSize; i++) {
        for (int j = 0; j < beamSize; j++) {
            State& state = states[i * beamSize + j];

            /* we push the incomplete hypothesis into the heap */
            if (fullHypos[state.pid].Count() == 0) {
                fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
            }
            else {
                HeapNode<float> node = fullHypos[state.pid].Top();
                float score = node.value;
                if (score < state.modelScore)
                    fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
            }
        }
    }
}

/*
save the output sequences in a tensor
>> output - output sequences (for return)
>> score - score of thes sequences
*/
void BeamSearch::Dump(IntList** output, XTensor* score)
{
    int dims[3] = { batchSize, 1 };

    InitTensor(score, 2, dims, X_FLOAT, -1);
    score->SetZeroAll();

    /* heap for an input sentence in the batch */
    for (int h = 0; h < batchSize; h++) {
        IntList* tgt = output[h];
        XHeap<MIN_HEAP, float>& heap = fullHypos[h];
        int c = heap.Count();

        float bestScore = -2e4F;
        State* state = NULL;
        for (int i = 0; i < c; i++) {
            auto node = heap.Pop();
            State* s = (State*)node.index;
            if (i == 0 || bestScore < node.value) {
                state = s;
                bestScore = node.value;
            }
        }

        int count = 0;
        bool isCompleted = true;

        /* we track the state from the end to the beginning */
        while (state != NULL) {
            if (!state->isCompleted)
                isCompleted = false;
            if (!isCompleted) {
                tgt->Add(state->prediction);
            }
            state = state->last;
        }
        tgt->Reverse();

        score->Set2D(bestScore, h, 0);
    }
}

/*
check if the token is an end symbol
>> token - token to be checked
*/
bool BeamSearch::IsEnd(int token)
{
    CheckNTErrors(endSymbolNum > 0, "No end symbol?");

    for (int i = 0; i < endSymbolNum; i++) {
        if (endSymbols[i] == token)
            return true;
    }

    return false;
}

/*
set end symbols for search
>> tokens - end symbols
>> tokenNum - number of the end symbols
*/
void BeamSearch::SetEnd(const int* tokens, const int tokenNum)
{
    if (endSymbols != NULL)
        delete[] endSymbols;

    if (tokenNum <= 0)
        return;

    /* we may have multiple end symbols */
    tokens = new int[tokenNum];
    for (int i = 0; i < tokenNum; i++)
        endSymbols[i] = tokens[i];
    endSymbolNum = tokenNum;
}

/*
check whether all hypotheses are completed
>> beam - the beam that keeps the searching states
*/
bool BeamSearch::IsAllCompleted(StateBundle* beam)
{
    State* states = beam->states;

    for (int i = 0; i < beam->stateNum; i++) {
        State& state = states[i];
        if (!state.isCompleted)
            return false;
    }

    return true;
}

/*
collect alive beam states
>> beam - the beam that keeps the searching states
<< aliveStates - the indices of unfinished states
*/
XTensor BeamSearch::GetAliveStates(StateBundle* beam)
{
    XTensor aliveStates;
    State* states = beam->states;

    /* get the indices of alive beam states */
    int count = 0;
    int* aliveStateList = new int[beam->stateNum];
    for (int i = 0; i < beam->stateNum; i += beamSize) {
        
        bool isHypoAllFinished = true;
        
        for (int j = 0; j < beamSize; j++) {
            if (!states[i + j].isCompleted)
                isHypoAllFinished = false;
        }

        if (!isHypoAllFinished) {
            for (int j = 0; j < beamSize; j++) {
                aliveStateList[count++] = i + j;
            }
        }
    }

    if (count == beam->stateNum) {
        delete[] aliveStateList;
        return aliveStates;
    }

    needReorder = true;
    InitTensor1D(&aliveStates, count, X_INT, beam->prediction.devID);
    aliveStates.SetData(aliveStateList, count);

    delete[] aliveStateList;
    return aliveStates;
}

/*
make a mask to prevent duplicated entries in beam expansion for the first position
>> beam - the beam that keeps the searching states
*/
XTensor BeamSearch::MakeFirstMask(StateBundle* beam)
{
    XTensor& prob = beam->probPath;
    XTensor mask;

    int order = prob.order;
    int dims[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < order - 1; i++)
        dims[i] = prob.dimSize[i];

    InitTensor(&mask, order - 1, dims, prob.dataType);
    mask.SetZeroAll();

    for (int i = 0; i < mask.unitNum; i++) {
        if (i % beamSize != 0)
            mask.Set(-2e4, i);
    }

    mask.FlushToDevice(prob.devID);

    return mask;
}

/* constructor */
GreedySearch::GreedySearch()
{
    maxLen = 0;
    batchSize = 0;
    endSymbolNum = 0;
    endSymbols = new int[32];
    startSymbol = -1;
    scalarMaxLength = -1;
}

/* de-constructor */
GreedySearch::~GreedySearch()
{
    if (endSymbols != NULL)
        delete[] endSymbols;
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void GreedySearch::Init(NMTConfig& config)
{
    maxLen = config.translation.maxLen;
    batchSize = config.common.sBatchSize;
    endSymbols[0] = config.model.eos;
    startSymbol = config.model.sos;
    scalarMaxLength = config.translation.maxLenAlpha;

    if (endSymbols[0] >= 0)
        endSymbolNum = 1;
}

/*
prepare for search
>> batchSize - size of the batch
*/
void GreedySearch::Prepare(int myBatchSize)
{
    batchSize = myBatchSize;
}

/* check if the token is an end symbol */
bool GreedySearch::IsEnd(int token)
{
    CheckNTErrors(endSymbolNum > 0, "No end symbol?");

    for (int i = 0; i < endSymbolNum; i++) {
        if (endSymbols[i] == token)
            return true;
    }

    return false;
}

/* set end symbols for search */
void GreedySearch::SetEnd(const int* tokens, const int tokenNum)
{
    if (endSymbols != NULL)
        delete[] endSymbols;

    if (tokenNum <= 0)
        return;

    /* we may have multiple end symbols */
    tokens = new int[tokenNum];
    for (int i = 0; i < tokenNum; i++)
        endSymbols[i] = tokens[i];
    endSymbolNum = tokenNum;
}

/*
search for the most promising states
>> model - the transformer model
>> input - input of the model
>> padding - padding of the input
>> outputs - outputs tokens of the search results
*/
void GreedySearch::Search(NMTModel* model, XTensor& input, 
                          XTensor& padding, IntList** outputs)
{
    XTensor maskEnc;
    XTensor encoding;
    batchSize = input.GetDim(0);

    input.SetDevice(model->devID);
    padding.SetDevice(model->devID);

    /* encoder mask */
    model->MakeMTMaskEnc(padding, maskEnc);

    /* make the encoding network */
    if (model->config->model.encPreLN)
        encoding = model->encoder->RunFastPreNorm(input, &maskEnc);
    else
        encoding = model->encoder->RunFastPostNorm(input, &maskEnc);
        
    /* max output-length = scalar * source-length */
    int lengthLimit = int(float(input.GetDim(-1)) * scalarMaxLength) + maxLen;

    CheckNTErrors(lengthLimit > 0, "Invalid maximum output length");

    /* the first token */
    XTensor inputDec;
    InitTensor2D(&inputDec, batchSize, 1, X_INT, input.devID);
    inputDec.SetDataFixed(startSymbol);

    /* initialize the finished flags */
    int* finishedFlags = new int[batchSize];
    for (int i = 0; i < batchSize; i++)
        finishedFlags[i] = 0;

    XTensor prob;
    XTensor maskEncDec;
    XTensor decoding;
    XTensor indexCPU;
    XTensor bestScore;

    InitTensorOnCPU(&indexCPU, &inputDec);
    InitTensor2D(&bestScore, batchSize, 1, encoding.dataType, encoding.devID);

    /* decoder mask */
    maskEncDec = model->MakeMTMaskDecInference(padding);

    for (int l = 0; l < lengthLimit; l++) {

        /* make the decoding network */
        if (model->config->model.decPreLN)
            decoding = model->decoder->RunFastPreNorm(inputDec, encoding, &maskEncDec, l);
        else
            decoding = model->decoder->RunFastPostNorm(inputDec, encoding, &maskEncDec, l);

        /* generate the output probabilities */
        prob = model->outputLayer->Make(decoding, false);

        /* get the most promising predictions */
        prob.Reshape(prob.dimSize[0], prob.dimSize[prob.order - 1]);
        TopK(prob, bestScore, inputDec, -1, 1);

        /* save the predictions */
        CopyValues(inputDec, indexCPU);

        for (int i = 0; i < batchSize; i++) {
            if (IsEnd(indexCPU.GetInt(i)))
                finishedFlags[i] = 1;
            else if (finishedFlags[i] != 1)
                (outputs[i])->Add(indexCPU.GetInt(i));
        }

        int finishedSentNum = 0;
        for (int i = 0; i < batchSize; i++)
            finishedSentNum += finishedFlags[i];
        if (finishedSentNum == batchSize) {
            l = lengthLimit;
            break;
        }
    }

    delete[] finishedFlags;
}

} /* end of the nmt namespace */