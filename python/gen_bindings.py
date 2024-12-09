import datetime
import re

OUTPUT_FILE: str = 'pccl/_cdecls.py'

HEADERS = [
    '../ccoip/public_include/ccoip_inet.h',
    '../include/pccl.h',
]

def comment_replacer(match):
    s = match.group(0)
    if s.startswith('/'):
        return ' '
    else:
        return s

def keep_line(line: str) -> bool:
    if line == '' or line.startswith('#'):
        return False
    if line.startswith('inline'):
        return False
    if line.startswith('extern "C"'):
        return False
    return True

pattern = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE
)

c_input: list[str] = []
for file in HEADERS:
    with open(file, 'rt') as f:
        full_src: str = f.read()
    full_src = re.sub(pattern, comment_replacer, full_src)  # remove comments
    data = [line.strip() for line in full_src.splitlines()]  # remove empty lines
    data = [line for line in data if keep_line(line)]  # remove empty lines
    data = [line.replace('PCCL_EXPORT', '') for line in data]
    for line in reversed(data):
        if line == '}':
            data.remove(line)
            break
    c_input += data

out = f'# Autogenered by {__file__} {datetime.datetime.now()}, do NOT edit!\n\n'
out += "__PCCL_CDECLS: str = '''\n\n"
for line in c_input:
    out += f'{line}\n'
out += "'''\n\n"

with open(OUTPUT_FILE, 'wt') as f:
    f.write(out)
