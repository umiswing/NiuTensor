#!/bin/sh
set -e

HARDWARE=$1
TASK=$2
SPLIT=2000

cat | parallel --pipe -L 4000 --keep-order "./nts-scripts/mosestokenizer -N -U en -c ./nts-scripts/nonbreaking_prefixes" > nts.tmp.tok 

# Apply BPE
./moses/fastbpe applybpe nts.tmp.bpe nts.tmp.tok ./model/bpe.code

# Translate
if [ "$HARDWARE" == "GPU" ]; then
    ./bin/NiuTensor -dev 0 -fp16 1 -model ./model/model.fp16 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt -sbatch 3072 -wbatch 64000 < nts.tmp.bpe | parallel --pipe -L 4000 --keep-order "perl ./nts-scripts/detokenizer.perl -q -l de"
else
    <nts.tmp.bpe parallel --pipe -L $SPLIT -N1 --keep-order "MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bin/NiuTensor -dev -1 -sbatch 64 -wbatch 2048 -model ./model/model.fp32 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt | perl ./nts-scripts/detokenizer.perl -q -l de"
fi

rm -rf nts.tmp*