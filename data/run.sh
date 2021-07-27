#!/bin/sh
set -e

HARDWARE=$1
TASK=$2
SPLIT=2000

# Pre-process
# cat | parallel --pipe -L 4000 -N1 --keep-order "perl ./moses/normalize-punctuation.perl -l en | perl ./moses/tokenizer.perl -q -l en -no-escape " > nts.tmp.tok 

cat | parallel --pipe -L 4000 --keep-order "./mosestokenizer -N -U en -c ./fast-mosestokenizer-master/share" > nts.tmp.tok 

# Apply BPE
./moses/fastbpe applybpe nts.tmp.bpe nts.tmp.tok ./model/bpe.code

# Translate
if [ "$HARDWARE" == "GPU" ]; then
    ./bin/NiuTensor -dev 6 -fp16 1 -model ./model/model.fp16 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt -sbatch 3072 -wbatch 64000 < nts.tmp.bpe | parallel --pipe -L 4000 --keep-order "perl ./moses/detokenizer.perl -q -l de"
    # ./bin/NiuTensor -dev 7 -fp16 1 -model ./model/model.fp16 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt < nts.tmp.bpe | parallel --pipe -L 4000 --keep-order "mosestokenizer -D de -c ./fast-mosestokenizer-master/share"
else
    total=`awk 'END{print NR}' nts.tmp.bpe`
    lines=`expr $total / $NUM_THREAD + 1`
    <nts.tmp.bpe parallel --pipe -L $SPLIT -N1 --keep-order "MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bin/NiuTensor -dev -1 -wbatch 5120 -model ./model/model.fp32 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt | perl ./moses/detokenizer.perl -q -l de"
fi

rm -rf nts.tmp*