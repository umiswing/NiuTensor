#!/bin/sh
set -e
num_processes=8

HARDWARE=$1
TASK=$2

# Pre-process
cat | parallel --pipe -L 5000 -N1 --keep-order "perl ./moses/normalize-punctuation.perl -l en | perl ./moses/tokenizer.perl -q -l en -no-escape -lines 5000" > nts.tmp.tok 

# Apply BPE
./moses/fastbpe applybpe nts.tmp.bpe nts.tmp.tok ./model/bpe.code

# Translate
if [ "$HARDWARE" == "GPU" ]; then
    ./bin/NiuTensor -dev 7 -fp16 1 -model ./model/model.fp16 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt < nts.tmp.bpe | parallel --pipe -L 5000 -N1 --keep-order "sed -r 's/(@@ )|(@@ ?$)//g' | perl ./moses/detokenizer.perl -q -l de"
else
    ./bin/NiuTensor -dev -1 -model ./model/model.fp32 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt < nts.tmp.bpe | parallel --pipe -L 5000 -N1 --keep-order "sed -r 's/(@@ )|(@@ ?$)//g' | perl ./moses/detokenizer.perl -q -l de"
fi

rm -rf nts.tmp*