import os
import sys
import json
import random

source_data_dir = str(sys.argv[1])
target_data_dir = str(sys.argv[2])
token = str(sys.argv[3])

with open(os.path.join(source_data_dir, token+'.json'), 'r') as f:
    source_unit_map = json.load(f)
with open(os.path.join(target_data_dir, token+'.json'), 'r') as f:
    target_unit_map = json.load(f)

orig_num = len(source_unit_map)
source_unit_map['<blank>'] = orig_num
idx = orig_num + 1
for unit in target_unit_map:
    if unit not in source_unit_map:
        source_unit_map[unit] = idx
        idx += 1

with open(os.path.join(target_data_dir, token+'_lp.json'), 'w', encoding='utf8') as f:
    json.dump(source_unit_map, f, indent=4, ensure_ascii=False)

with open(os.path.join(target_data_dir, token+'_lp.txt'), 'w', encoding='utf8') as f:
    for unit in source_unit_map:
        f.write(unit)
        f.write('\n')
    f.write('|')