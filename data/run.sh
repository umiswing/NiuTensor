#!/bin/sh
set -e

NUM_THREAD=18
HARDWARE=$1
TASK=$2

# Pre-process
cat | parallel --pipe -L 4000 -N1 --keep-order "perl ./moses/normalize-punctuation.perl -l en | perl ./moses/tokenizer.perl -q -l en -no-escape " > nts.tmp.tok 

# Apply BPE
./moses/fastbpe applybpe nts.tmp.bpe nts.tmp.tok ./model/bpe.code

# Translate
if [ "$HARDWARE" == "GPU" ]; then
    ./bin/NiuTensor -dev 7 -fp16 1 -model ./model/model.fp16 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt < nts.tmp.bpe | parallel --pipe -L 4000 -N1 --keep-order "sed -r 's/(@@ )|(@@ ?$)//g' | perl ./moses/detokenizer.perl -q -l de"
else
    total=`awk 'END{print NR}' nts.tmp.bpe`
    lines=`expr $total / $NUM_THREAD + 1`
    <nts.tmp.bpe parallel --pipe -L $lines -N1 --keep-order "./bin/NiuTensor -dev -1 -model ./model/model.fp32 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt | sed -r 's/(@@ )|(@@ ?$)//g' | perl ./moses/detokenizer.perl -q -l de"
fi

rm -rf nts.tmp*