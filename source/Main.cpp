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
 * $Created by: Chi Hu (huchinlp@gmail.com) 2021-11-06
 */

#include <iostream>

#include "./nmt/Config.h"
#include "./nmt/train/Trainer.h"
#include "./nmt/translate/Translator.h"

using namespace nmt;

int main(int argc, const char** argv)
{
    std::clock_t mainStart = std::clock();

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

    /* translation */
    else if (strcmp(config.translation.inputFN, "") != 0) {

        /* disable gradient flow */
        DISABLE_GRAD;

        NMTModel model;
        model.InitModel(config);
        model.SetTrainingFlag(false);

        Translator translator;
        translator.Init(config, model);
        translator.Translate();

        BeamSearch* searcher = (BeamSearch*)(translator.seacher);
        LOG("Duration of encoding: %f", searcher->encoderCost);
        LOG("Duration of decoding: %f", searcher->decoderCost);
        LOG("Duration of decoder selfAttnCost: %f", model.decoder->selfAttnCost);
        LOG("Duration of decoder endeAttnCost: %f", model.decoder->endeAttnCost);
        LOG("Duration of decoder lnCost: %f", model.decoder->lnCost);
        LOG("Duration of decoder ffnCost: %f", model.decoder->ffnCost);
        LOG("Duration of decoder embCost: %f", model.decoder->embCost);

        LOG("Duration of output: %f", searcher->outputCost);
        LOG("Duration of caching: %f", searcher->cachingCost);
        LOG("Duration of beam search: %f", searcher->beamSearchCost);
        LOG("Duration of scoring: %f", searcher->scoringCost);
        LOG("Duration of generating: %f", searcher->generatingCost);
        LOG("Duration of expanding: %f", searcher->expandingCost);
        LOG("Duration of collecting: %f", searcher->collectingCost);
    }

    else {
        fprintf(stderr, "Thanks for using NiuTrans.NMT! This is an effcient\n");
        fprintf(stderr, "neural machine translation system. \n\n");
        fprintf(stderr, "   Run this program with \"-train\" for training!\n");
        fprintf(stderr, "Or run this program with \"-input\" for translation!\n");
    }

    /*XTensor emb, idx;
    InitTensor2D(&emb, 30000, 512, X_FLOAT, 0);
    InitTensor2D(&idx, 512, 1, X_INT, 0);
    int data[512];
    for (int i = 0; i < 512; i++)
        data[i] = rand() % 30000;
    idx.SetData(data, 512);
    XTensor x;
    for (int i = 0; i < 100000; i++)
        x = Gather(emb, idx);

    LOG("Duration of main: %f", (std::clock() - mainStart) / (double)CLOCKS_PER_SEC);*/

    return 0;
}