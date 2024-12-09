import datetime
import re

OUTPUT_FILE: str = 'pccl/_cdecls.py'

HEADERS = [
    '../include/pccl.h',
    '../include/pccl_status.h',
    '../ccoip/public_include/ccoip_inet.h'
]

def comment_replacer(match):
    s = match.group(0)
    if s.startswith('/'):
        return ' '
    else:
        return s


enums_names: list[str] = []
struct_names: list[str] = []


def keep_line(line: str) -> bool:
    if line == '' or line.startswith('#'):
        return False
    if line.startswith('PCCL_EXPORT') and not line.startswith('extern "C"'):
        return True
    if line.startswith('typedef enum'):
        enums_names.append(line.split()[2])
        return False
    if line.startswith('typedef struct'):
        struct_names.append(line.split()[2])
        return False
    if line.startswith('typedef'):
        return True
    return False

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
    c_input += data

out = f'# Autogenered by {__file__} {datetime.datetime.now()}, do NOT edit!\n\n'
out += "__PCCL_CDECLS: str = '''\n\n"
for struct in struct_names:
    out += f'typedef struct {struct} {struct};\n'
out += '\n'
for enum in enums_names:
    out += f'typedef int {enum};\n'
out += '\n'
for line in c_input:
    out += f'{line}\n'
out += "'''\n\n"

with open(OUTPUT_FILE, 'wt') as f:
    f.write(out)
