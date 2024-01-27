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

for unit in target_unit_map:
    if unit not in source_unit_map:
        u = random.choice(list(source_unit_map.keys()))
        while u in target_unit_map:
            u = random.choice(list(source_unit_map.keys()))
        source_unit_map[unit] = source_unit_map[u]
        del source_unit_map[u]

with open(os.path.join(target_data_dir, token+'_lp.json'), 'w', encoding='utf8') as f:
    json.dump(source_unit_map, f, indent=4, ensure_ascii=False)

with open(os.path.join(target_data_dir, token+'_lp.txt'), 'w', encoding='utf8') as f:
    for unit, idx in source_unit_map.items():
        f.write(unit)
        f.write('\n')
    f.write('<blank>\n')
    f.write('|')
