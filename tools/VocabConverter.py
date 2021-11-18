'''
Convert a fairseq vocab to a NiuTrans.NMT vocab
Usage: python3 VocabConverter.py -raw [fairseq_vocab] -new [niutrans_nmt_vocab]
'''

import sys
import argparse

parser = argparse.ArgumentParser(
    description='Convert a fairseq vocabulary to a NiuTrans.NMT vocabulary')
parser.add_argument(
    '-raw', help='Path of the fairseq vocabulary', type=str, default='')
parser.add_argument(
    '-new', help='Path of the NiuTrans.NMT vocabulary to be saved', type=str, default='')
args = parser.parse_args()

# User defined words
PAD=1
SOS=2
EOS=2
UNK=3

with open(args.raw, "r", encoding="utf8") as fi:
    with open(args.new, "w", encoding="utf8") as fo:
        lines = fi.readlines()

        # the first several indices are reserved
        start_id = UNK + 1
        
        # the first line: vocab_size, start_id
        fo.write("{} {}\n".format(len(lines)+start_id, start_id))

        # other lines: word, id
        for l in lines:
            fo.write("{} {}\n".format(l.split()[0], start_id))
            start_id += 1