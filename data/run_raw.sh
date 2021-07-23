#!/bin/sh
set -e
num_processes=8

HARDWARE=$1
TASK=$2

# Pre-process
cat <&0 > nts.tmp.raw
total=`awk 'END{print NR}' nts.tmp.raw`
lines=`expr $total / $num_processes + 1`
split -l $lines nts.tmp.raw -d -a 1 nts.tmp.raw.
for ((i=0;i<$num_processes;i++)); do
{
    perl ./moses/normalize-punctuation.perl -l en < nts.tmp.raw.$i > nts.tmp.norm.$i
} &
done
wait
for ((i=0;i<$num_processes;i++))do echo nts.tmp.norm.$i;done | xargs -i cat {} >> nts.tmp.norm
perl ./moses/tokenizer.perl -q -l en -threads 10 -no-escape -lines 10000 < nts.tmp.norm > nts.tmp.tok 

# Apply BPE
./moses/fastbpe applybpe nts.tmp.bpe nts.tmp.tok ./model/bpe.code

# Translate
if [ "$HARDWARE" == "GPU" ]; then
    ./bin/NiuTensor -dev 7 -fp16 1 -model ./model/model.fp16 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt < nts.tmp.bpe > nts.tmp.atat
else
    ./bin/NiuTensor -dev -1 -model ./model/model.fp32 -srcvocab ./model/vocab.txt -tgtvocab ./model/vocab.txt < nts.tmp.bpe > nts.tmp.atat
fi

# Remove BPE
sed -r 's/(@@ )|(@@ ?$)//g' < nts.tmp.atat > nts.tmp

# Split ouput file
total=`awk 'END{print NR}' nts.tmp`
lines=`expr $total / $num_processes + 1`
split -l $lines nts.tmp -d -a 1 nts.tmp.

# Run detokenizing in parallel
for ((i=0;i<$num_processes;i++)); do
{
    perl ./moses/detokenizer.perl -q -l de < nts.tmp.$i > nts.tmp.output.$i
} &
done
wait

for ((i=0;i<$num_processes;i++))do echo nts.tmp.output.$i;done | xargs -i cat {}

rm -rf nts.tmp*