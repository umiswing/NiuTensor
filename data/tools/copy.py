import sys

with open(sys.argv[1], 'r', encoding='utf8') as fi:
    with open(sys.argv[2], 'w', encoding='utf8') as fo:
        content = fi.read()
        for i in range(33):
            fo.write(content)