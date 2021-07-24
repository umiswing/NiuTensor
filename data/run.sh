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
    total=`awk 'END{print NR}' nts.tmp.bpe`
    lines=`expr $total / $num_processes + 1`
    num=`expr $total / $lines`
    if [ "$num_processes" -gt "$num" ]; then
        num_processes=$num
    fi
    split -l $lines nts.tmp.bpe -d -a 1 nts.tmp.bpe.
    for ((i=0;i<$num_processes;i++)); do
    {
        ./bin/NiuTensor -dev -1 -model ./model/model.fp32 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt < nts.tmp.bpe.$i | parallel --pipe -L 5000 -N1 --keep-order "sed -r 's/(@@ )|(@@ ?$)//g' | perl ./moses/detokenizer.perl -q -l de" > nts.tmp.res.$i
    } &
    done
    wait
    for ((i=0;i<$num_processes;i++))do echo nts.tmp.res.$i;done | xargs -i cat {}
fi

rm -rf nts.tmp*