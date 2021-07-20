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
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#include <fstream>
#include "Config.h"

using namespace nts;
using namespace std;

/* the nmt (NiuTrans.NMT) namespace */
namespace nmt
{

/*
load configurations from the command
>> argc - number of arguments
>> argv - the list of arguments
*/
NMTConfig::NMTConfig(int argc, const char** argv)
{
    char** args = new char* [MAX_PARAM_NUM];
    for (int i = 0; i < argc; i++) {
        args[i] = new char[strlen(argv[i]) + 1];
        strcpy(args[i], argv[i]);
    }
    for (int i = argc; i < MAX_PARAM_NUM; i++) {
        args[i] = NULL;
    }

    char* configFN = new char[1024];
    LoadParamString(argc, args, "config", configFN, "");

    int argsNum = argc;

    /* override the configuration according to the file content */
    if (strcmp(configFN, "") != 0)
        argsNum = LoadFromFile(configFN, args);

    ShowParams(argsNum, args);

    /* parse configuration in args */
    model.Load(argsNum, (const char **)args);
    common.Load(argsNum, (const char **)args);
    training.Load(argsNum, (const char **)args);
    translation.Load(argsNum, (const char **)args);

    for (int i = 0; i < MAX(argc, argsNum); i++)
        delete[] args[i];
    delete[] args;
    delete[] configFN;
}

/*
load configurations from a file
>> configFN - path to the configuration file
>> args - the list to store the configurations
<< argsNum - the number of arguments
format: one option per line, separated by a blank or a tab
*/
int NMTConfig::LoadFromFile(const char* configFN, char** args) 
{
    ifstream f(configFN, ios::in);
    CheckNTErrors(f.is_open(), "Failed to open the config file");

    int argsNum = 0;

    /* parse arguments from the file */
    string key, value;
    while (f >> key >> value && argsNum < (MAX_PARAM_NUM - 1)) {
        if (args[argsNum] != NULL) {
            delete[] args[argsNum];
        }
        if (args[argsNum + 1] != NULL) {
            delete[] args[argsNum + 1];
        }
        args[argsNum] = new char[1024];
        args[argsNum + 1] = new char[1024];
        strcpy(args[argsNum++], key.c_str());
        strcpy(args[argsNum++], value.c_str());
    }

    /* record the number of arguments */
    return argsNum;
}

/* load model configuration from the command */
void ModelConfig::Load(int argsNum, const char** args)
{
    Create(argsNum, args);
    LoadBool("usebigatt", &useBigAtt, false);
    LoadBool("decoderonly", &decoderOnly, false);
    LoadBool("enchistory", &useEncHistory, false);
    LoadBool("dechistory", &useDecHistory, false);
    LoadInt("srcvocabsize", &srcVocabSize, -1);
    LoadInt("maxsrc", &maxSrcLen, 200);
    LoadInt("encheads", &encSelfAttHeadNum, -1);
    LoadInt("encemb", &encEmbDim, -1);
    LoadInt("encffn", &encFFNHiddenDim, -1);
    LoadInt("enclayer", &encLayerNum, -1);
    LoadBool("encprenorm", &encPreLN, false);
    LoadInt("tgtvocabsize", &tgtVocabSize, -1);
    LoadInt("maxtgt", &maxTgtLen, -1);
    LoadInt("decheads", &decSelfAttHeadNum, -1);
    LoadInt("decemb", &decEmbDim, -1);
    LoadInt("decffn", &decFFNHiddenDim, -1);
    LoadInt("declayer", &decLayerNum, -1);
    LoadBool("decprenorm", &decPreLN, false);
    LoadInt("maxrp", &maxRelativeLength, -1);
    LoadInt("pad", &pad, -1);
    LoadInt("unk", &unk, -1);
    LoadInt("sos", &sos, -1);
    LoadInt("eos", &eos, -1);
    LoadBool("encfinalnorm", &encFinalNorm, true);
    LoadBool("decfinalnorm", &decFinalNorm, true);
    LoadBool("shareallemb", &shareEncDecEmb, false);
    LoadBool("sharedec", &shareDecInputOutputEmb, false);
    LoadFloat("dropout", &dropout, 0.0F);
    LoadFloat("ffnropout", &ffnDropout, 0.0F);
    LoadFloat("attdropout", &attDropout, 0.0F);
}

/* load training configuration from the command */
void TrainingConfig::Load(int argsNum, const char **args)
{
    Create(argsNum, args);
    LoadString("train", trainFN, "");
    LoadString("valid", validFN, "");
    isTraining = (strcmp(trainFN, "") == 0) ? false : true;
    LoadFloat("lrate", &lrate, 0.0015F);
    LoadFloat("lrbias", &lrbias, 0);
    LoadInt("nepoch", &nepoch, 50);
    LoadInt("nstep", &nstep, 100000);
    LoadInt("updatefreq", &updateFreq, 1);
    LoadInt("savefreq", &saveFreq, 1);
    LoadInt("ncheckpoint", &ncheckpoint, 10);
    LoadInt("nwarmup", &nwarmup, 8000);
    LoadBool("adam", &useAdam, true);
    LoadFloat("adambeta1", &adamBeta1, 0.9F);
    LoadFloat("adambeta2", &adamBeta2, 0.98F);
    LoadFloat("adamdelta", &adamDelta, 1e-9F);
    LoadFloat("labelsmoothing", &labelSmoothingP, 0.1F);
}

/* load training configuration from the command */
void TranslationConfig::Load(int argsNum, const char** args)
{
    Create(argsNum, args);
    LoadString("input", inputFN, "");
    LoadString("output", outputFN, "");
    LoadInt("maxlen", &maxLen, 200);
    LoadInt("beam", &beamSize, 1);
    LoadFloat("lenalpha", &lenAlpha, 0.F);

    /* smaller value may leads to worse translations but higher speed */
    LoadFloat("maxlenalpha", &maxLenAlpha, 1.25F);
}

/* load training configuration from the command */
void CommonConfig::Load(int argsNum, const char** args)
{
    Create(argsNum, args);
    LoadString("model", modelFN, "");
    LoadString("srcvocab", srcVocabFN, "");
    LoadString("tgtvocab", tgtVocabFN, "");
    LoadInt("seed", &seed, 1);
    LoadInt("dev", &devID, -1);
    LoadInt("seed", &seed, 1);
    LoadInt("loginterval", &logInterval, 100);
    LoadInt("wbatch", &wBatchSize, 40960);
    LoadInt("sbatch", &sBatchSize, 768);
    LoadInt("bufsize", &bufSize, 2000000);
    LoadInt("bucketsize", &bucketSize, -1);
    LoadBool("fp16", &useFP16, false);
}

/* 
split string into sub-strings by a delimiter
>> s - the original string
>> delimiter - as it is
>> maxNum - the maximum number of sub-strings
<< substrings - all sub-strings
*/
vector<string> SplitString(const string& s, const string& delimiter, int maxNum)
{
    CheckNTErrors(delimiter.length() > 0, "Invalid delimiter");
    
    vector<string> substrings;
    size_t pos = 0;
    size_t start = 0;
    while ((pos = s.find(delimiter, start)) != string::npos) {
        if (pos != start) {
            substrings.emplace_back(s.substr(start, pos - start));
        }
        start = pos + delimiter.length();
        if (substrings.size() == maxNum)
            break;
    }
    if (start != s.length() && substrings.size() < maxNum) {
        substrings.emplace_back(s.substr(start, s.length() - start));
    }
    return substrings;
}

} /* end of the nmt (NiuTrans.NMT) namespace */