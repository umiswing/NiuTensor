#ifdef WIN32
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#endif
#else
#define DBG_NEW new
#endif
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

#ifdef WIN32
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif // WIN32

#include <stdio.h>
#include "./network/XNet.h"
#include "./tensor/XUtility.h"
#include "./tensor/function/FHeader.h"
#include "./tensor/core/CHeader.h"
#include "./tensor/test/Test.h"
#include "./sample/fnnlm/FNNLM.h"
#include "./sample/transformer/NMT.h"
#include "./train/TTrain.h"

using namespace nts;
using namespace fnnlm;
using namespace nmt;

void test(int argc, const char** argv) {
    XConfig config;

    if (argc > 1) {
        config.Create(argc - 1, argv + 1);
        verboseLevel = config.GetInt("verbose", 1);
    }

    if (argc > 1 && !strcmp(argv[1], "-test"))
        Test();
    else if (argc > 1 && !strcmp(argv[1], "-testtrain"))
        TestTrain();
    else if (argc > 1 && !strcmp(argv[1], "-fnnlm"))
        FNNLMMain(argc - 1, argv + 1);
    else if (argc > 1 && !strcmp(argv[1], "-nmt"))
        NMTMain(argc - 1, argv + 1);
    else {
        fprintf(stderr, "Thanks for using NiuTensor! This is a library for building\n");
        fprintf(stderr, "neural networks in an easy way. \n\n");
        fprintf(stderr, "   Run this program with \"-test\" for unit test!\n");
        fprintf(stderr, "Or run this program with \"-testtrain\" for test of the trainer!\n");
        fprintf(stderr, "Or run this program with \"-fnnlm\" for sample FNNLM!\n");
        fprintf(stderr, "Or run this program with \"-nmt\" for sample Transformer!\n");
    }
}

void test2() {
    DISABLE_GRAD;
    XTensor a;
    XTensor b;
    InitTensor2D(&a, 2, 2, X_FLOAT, 0);
    InitTensor2D(&b, 2, 2, X_FLOAT, 0);
    a.SetDataFixed(1.0F);
    b = ConvertDataType(a, X_FLOAT16);
    
    for (int i = 0; i < 4; i++) {
        a = MatrixMul(a, a);
        b = MatrixMul(b, b);
    }
    XTensor c = ConvertDataType(b, X_FLOAT);
    a.Dump(stderr, "a");
    c.Dump(stderr, "c");
}

int main(int argc, const char ** argv)
{

#ifdef WIN32
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif // WIN32

    //test2();
    test(argc, argv);

#ifdef WIN32
    _CrtDumpMemoryLeaks();
#endif // WIN32

    return 0;
}
