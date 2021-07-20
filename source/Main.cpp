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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-10
 */

#include <iostream>
#include "tensor/core/CHeader.h"
#include "tensor/function/Softmax.h"
#include "./sample/transformer/NMT.h"
#include "tensor/test/Test.h"

using namespace nts;
using namespace nmt;

void test() {
    XTensor a, b, c, d;
    InitTensor2D(&a, 2, 4, X_FLOAT, -1);

    DTYPE sData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE meanData[4] = { 2.0F, 3.0F, 4.0F, 5.0F };
    DTYPE answer[4] = { 2.0F, 2.0F, 2.0F, 2.0F };

    a.SetData(sData, 8);
    //a = ConvertDataType(a, X_FLOAT16);
    b = ReduceMean(a, 0);
    c = ReduceVariance(a, 0, b, true);
    c = ConvertDataType(c, X_FLOAT);
    c.Dump(stderr);
}

int main(int argc, const char ** argv)
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    NMTMain(argc, argv);

    return 0;
}