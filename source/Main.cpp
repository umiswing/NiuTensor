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

        if (config.translation.beamSize > 1) {
            BeamSearch* searcher = (BeamSearch*)(translator.seacher);
            LOG("Duration of beam search: %f", searcher->beamSearchCost);

            LOG("Duration of encoding: %.3f", searcher->encoderCost / searcher->beamSearchCost);
            LOG("Duration of decoding: %.3f", searcher->decoderCost / searcher->beamSearchCost);
            LOG("Duration of decoder selfAttnCost: %.3f", model.decoder->selfAttnCost / searcher->beamSearchCost);
            LOG("Duration of decoder endeAttnCost: %.3f", model.decoder->endeAttnCost / searcher->beamSearchCost);
            LOG("Duration of decoder lnCost: %.3f", model.decoder->lnCost / searcher->beamSearchCost);
            LOG("Duration of decoder ffnCost: %.3f", model.decoder->ffnCost / searcher->beamSearchCost);
            LOG("Duration of decoder embCost: %.3f", model.decoder->embCost / searcher->beamSearchCost);

            LOG("Duration of output: %.3f", searcher->outputCost / searcher->beamSearchCost);
            LOG("Duration of caching: %.3f", searcher->cachingCost / searcher->beamSearchCost);
            
            LOG("Duration of scoring: %.3f", searcher->scoringCost / searcher->beamSearchCost);
            LOG("Duration of generating: %.3f", searcher->generatingCost / searcher->beamSearchCost);
            LOG("Duration of expanding: %.3f", searcher->expandingCost / searcher->beamSearchCost);
            LOG("Duration of collecting: %.3f", searcher->collectingCost / searcher->beamSearchCost);
        }
        else {
            GreedySearch* searcher = (GreedySearch*)(translator.seacher);
            LOG("Duration of greedySearchCost: %f", searcher->greedySearchCost);
            LOG("Duration of encoding: %.2f", searcher->encoderCost / searcher->greedySearchCost);
            LOG("Duration of decoding: %.2f", searcher->decoderCost / searcher->greedySearchCost);
            LOG("Duration of outputCost: %.2f", searcher->outputCost / searcher->greedySearchCost);
            LOG("Duration of topKCost: %.2f", searcher->topKCost / searcher->greedySearchCost);
            LOG("Duration of copyCost: %.2f", searcher->copyCost / searcher->greedySearchCost);
        }
    }

    else {
        fprintf(stderr, "Thanks for using NiuTrans.NMT! This is an effcient\n");
        fprintf(stderr, "neural machine translation system. \n\n");
        fprintf(stderr, "   Run this program with \"-train\" for training!\n");
        fprintf(stderr, "Or run this program with \"-input\" for translation!\n");
    }

    LOG("Duration of main: %f", (std::clock() - mainStart) / (double)CLOCKS_PER_SEC);

    return 0;
}