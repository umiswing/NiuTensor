import os

ifiles = []

for path, subdirs, files in os.walk('./niutensor/source/'):
    for name in files:
        ifiles.append(os.path.join(path, name))

# for f in ifiles:
#     if '.cpp' in f:
#         with open(f, "r", encoding='utf8') as fi:
#             content = fi.read()
#             new_content = '#ifndef DBG_NEW\n#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )\n#endif\n'
#             new_content += content.replace('new', 'DBG_NEW')
#         with open(f, "w", encoding='utf8') as fo:
#             fo.write(new_content)

for f in ifiles:
    if '.cpp' in f:
        with open(f, "r", encoding='utf8') as fi:
            content = fi.read()
            new_content = content.replace('#ifndef DBG_NEW\n#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )\n#endif\n', '#ifdef WIN32\n#ifndef DBG_NEW\n#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )\n#endif\n#else\n#define DBG_NEW new\n#endif\n')
        with open(f, "w", encoding='utf8') as fo:
            fo.write(new_content)