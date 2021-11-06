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
 * $Created by: Chi Hu (huchinlp@gmail.com) 2021-11-06
 */

#include <iostream>

#include "./nmt/Config.h"
#include "./nmt/train/Trainer.h"
#include "./nmt/translate/Translator.h"

using namespace nmt;

int Main(int argc, const char** argv)
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc == 0)
        return 1;

    /* load configurations */
    NMTConfig config(argc, argv);

    srand(config.common.seed);

    /* training */
    if (strcmp(config.training.trainFN, "") != 0) {

        NMTModel model;
        model.InitModel(config);

        Trainer trainer;
        trainer.Init(config, model);
        trainer.Run();
    }

    /* translating */
    else {

        /* disable gradient flow */
        DISABLE_GRAD;

        NMTModel model;
        model.InitModel(config);

        Translator translator;
        translator.Init(config, model);
        translator.Translate();
    }

    return 0;
}