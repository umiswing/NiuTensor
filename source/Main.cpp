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

using namespace nts;
using namespace nmt;

void test() {
    for (int i = 0; i < 200; i++) {
        XTensor a, b, c, d;
        InitTensor3D(&a, 31, 1, 512, X_FLOAT, 0);
        a.SetDataRand();
        a = ConvertDataType(a, X_FLOAT16);
        InitTensor3D(&b, 31, 512, 1, X_FLOAT, 0);
        b.SetDataRand();
        b = ConvertDataType(b, X_FLOAT16);
        c = BMMul(a, b);
        c = ConvertDataType(c, X_FLOAT);
        c.Dump(stderr, "c");
    }
}

int main(int argc, const char ** argv)
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    //test();
    NMTMain(argc, argv);

    return 0;
}