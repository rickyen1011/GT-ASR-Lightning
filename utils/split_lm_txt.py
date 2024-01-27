import os
import sys
import json
import random

data_dir = str(sys.argv[1])
with open(os.path.join(data_dir, 'test.json'), 'r') as f:
    numWords = json.load(f)["number_of_words"]
with open(os.path.join(data_dir, 'vocab.txt'), 'r') as f:
    all_words = f.readlines()

start = 0
for lang, num in numWords.items():
    end = start + num
    output = open(os.path.join(data_dir, 'lm', lang, 'vocab.txt'), 'w')
    for i in range(start, end):
        output.write(all_words[i])
    
    start = end
