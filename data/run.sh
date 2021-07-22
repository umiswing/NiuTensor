set -e
num_processes=8

rm -rf res.txt output.txt

# Pre-process
echo "Normalizing & Tokenizing..."
total=`awk 'END{print NR}' $1`
lines=`expr $total / $num_processes + 1`
split -l $lines $1 -d -a 1 $1.nts.gpu.
for ((i=0;i<$num_processes;i++)); do
{
    perl ./tools/normalize-punctuation.perl -l en < $1.nts.gpu.$i > $1.nts.gpu.norm.$i
} &
done
wait
for ((i=0;i<$num_processes;i++))do echo $1.nts.gpu.norm.$i;done | xargs -i cat {} >> $1.nts.gpu.norm
perl ./tools/tokenizer.perl -l en -threads 8 -no-escape -lines 20000 < $1.nts.gpu.norm > $1.nts.gpu.norm.tok

echo "Applying BPE"
./tools/fastbpe applybpe $1.nts.gpu.bpe $1.nts.gpu.norm.tok en-de.code

# Translate
../bin/NiuTensor -dev 7 -fp16 1 -model ../data/model.fp16 -srcvocab ../data/vocab.en -tgtvocab ../data/vocab.en < $1.nts.gpu.bpe > $2.nts.gpu.atat

# Remove BPE
sed -r 's/(@@ )|(@@ ?$)//g' < $2.nts.gpu.atat > $2.nts.gpu.tok

# Split ouput file
total=`awk 'END{print NR}' $2.nts.gpu.tok`
lines=`expr $total / $num_processes + 1`
split -l $lines $2.nts.gpu.tok -d -a 1 $2.nts.gpu.tok.

# Run detokenizing in parallel
for ((i=0;i<$num_processes;i++)); do
{
    perl ./tools/detokenizer.perl -l de -a < $2.nts.gpu.tok.$i > $2.nts.gpu.$i
} &
done
wait

# Combine all results
for ((i=0;i<$num_processes;i++))do echo $2.nts.gpu.$i;done | xargs -i cat {} >> $2

rm -rf $1.nts.gpu* $2.nts.gpu*