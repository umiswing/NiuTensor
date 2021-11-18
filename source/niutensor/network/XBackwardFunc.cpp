/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
 * All rights reserved.
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
 * backward computation for activation function
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-18
 * Dingdang won 5 games in the GO training yesterday, hahaha ...
 */

#include "XNoder.h"
#include "XBackwardFunc.h"
#include "../tensor/XName.h"
#include "../tensor/function/FHeader.h"

namespace nts{

/* compute dE/dx of a node */
void XFuncGrad::MakeGrad(XTensor * node, bool isEfficient)
{
    if (!isEfficient) {
        CheckNTErrors(node->grad != NULL, "No gradient found!");
    }
    else {
        CheckNTErrors(!node->isGrad || node->grad != NULL, "No gradient found!");
    }

    XLink &income = node->income;
    int operID = income.typeID;

    CheckNTErrors(income.tailNum == 1, "Too many input tensors for the function!");

    XTensor * input = income.tails[0];
    XTensor * output = node;

    if (!isEfficient || input->isGrad) {
        /* the data address is lazy allocated */
        input->grad = NewTensor(input, false);

        XTensor * dedx = input->grad;
        XTensor * dedy = output->grad;

        /* share the data address in this case (need more tests!) */
        if (_IsSameShaped(input->grad, output->grad)) {
            input->grad->data = output->grad->data;
        }
        else {
            input->grad = NewTensor(input);
        }

        XTensor* tmp;

        /* store the result to a temporary node if the input has multiple children */
        if (input->outgo.tailNum > 1) {
            tmp = NewTensor(output);
            tmp->SetZeroAll();
        }
        /* otherwise, the result is directly stored into the input node  */
        else {
            tmp = dedx;
        }

        if (operID == FUNC_HARDTANH)
            _HardTanHBackward(output, input, dedy, tmp);
        else if (operID == FUNC_IDENTITY)
            _IdentityBackward(output, input, dedy, tmp);
        else if (operID == FUNC_LOGSOFTMAX) {
            int leadDim = income.GetParamInt(0);
            CheckNTErrors(leadDim >= 0 && leadDim < input->order, "wrong leading dimension in logsoftmax!");
            _LogSoftmaxBackward(NULL, output, input, dedy, tmp, NULL, leadDim, NOLOSS);
        }
        else if (operID == FUNC_RECTIFY) {
            _RectifyBackward(output, input, dedy, tmp);

            /* delete unused data to save memory */
            output->DestroyData();
        }
        else if (operID == FUNC_SIGMOID)
            _SigmoidBackward(output, input, dedy, tmp);
        else if (operID == FUNC_SOFTMAX) {
            int leadDim = income.GetParamInt(0);
            CheckNTErrors(leadDim >= 0 && leadDim < input->order, "wrong leading dimension in softmax!");
            _SoftmaxBackward(NULL, output, input, dedy, tmp, NULL, leadDim, NOLOSS);

            /* delete unused data to save memory */
            output->DestroyData();
            input->DestroyData();
        }
        else {
            ShowNTErrors("Unsupported backward computation! TODO!");
        }

        if (input->outgo.tailNum > 1) {
            _SumMe(dedx, tmp);
            DelTensor(tmp);
        }

        /* the data address is shared and the raw pointer should be NULL to escape re-delete */
        if (input->grad->data == output->grad->data) {
            output->grad->data = NULL;
        }
    }

    node->visitMark = NODE_FINISHED;
    node->isGradFinished = true;
}

/* indicates whether the node is for an activation function */
bool XFuncGrad::IsFunc(XTensor * node)
{
    XLink &income = node->income;
    return (income.typeID & FUNCTION_BASE) != 0;
}

}
