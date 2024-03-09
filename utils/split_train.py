import os
import sys
import json
import random

source_data_dir = str(sys.argv[1])
thresh = int(sys.argv[2])

with open(os.path.join(source_data_dir, 'train.json'), 'r') as f:
    all_train = json.load(f)
    total_number = list(all_train["number_of_samples"].values())[0]

output_dir = os.path.join(source_data_dir, f'subset{thresh}')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(10):
    subset = {}
    subset["languages"] = all_train["languages"]
    subset["dataset"] = all_train["dataset"]
    subset["number_of_words"] = 0
    subset["number_of_samples"] = 0
    subset["wordcounts"] = {}
    subset["filenames"] = {}

    count = 0
    d = all_train["filenames"]
    shuffle_filename = {k:d[k] for k in random.sample(list(d.keys()), len(d))}
    while count != thresh:
        count = 0
        subset = {}
        subset["languages"] = all_train["languages"]
        subset["dataset"] = all_train["dataset"]
        subset["number_of_words"] = 0
        subset["number_of_samples"] = 0
        subset["wordcounts"] = {}
        subset["filenames"] = {}
        for kw, filenames in shuffle_filename.items():
            for filepath in filenames:
                if random.random() > thresh / total_number  or count >= thresh:
                    continue
                if kw not in subset["wordcounts"]:
                    subset["wordcounts"][kw] = 0
                    subset["filenames"][kw] = []
                    subset["number_of_words"] += 1
                subset["wordcounts"][kw] += 1
                subset["filenames"][kw].append(filepath)
                subset["number_of_samples"] += 1
                count += 1
    
    with open(os.path.join(output_dir, f'train{i+1}.json'), 'w', encoding='utf8') as f:
        json.dump(subset, f, indent=4, ensure_ascii=False)