#!/bin/sh
set -e
num_processes=18

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
    lines=`expr $total / $num_processes + 1`
    num=`expr $total / $lines`
    if [ "$num_processes" -gt "$num" ]; then
        num_processes=$num
    fi
    split -l $lines nts.tmp.bpe -d -a 2 nts.tmp.bpe.
    num=`expr $num_processes - 1`
    for ((i=0;i<$num_processes;i++)); do
    {
        printf -v ii "%02d" $i
        ./bin/NiuTensor -dev -1 -model ./model/model.fp32 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt < nts.tmp.bpe.$ii | parallel --pipe -L 4000 -N1 --keep-order "sed -r 's/(@@ )|(@@ ?$)//g' | perl ./moses/detokenizer.perl -q -l de" > nts.tmp.res.$ii
    } &
    done
    wait
    cat nts.tmp.res.*
fi

rm -rf nts.tmp*