import sys

with open(sys.argv[1], 'r', encoding='utf8') as fi:
    lines = [l.count(' ') for l in fi]
    lines.sort()
    print(lines[-20:])
