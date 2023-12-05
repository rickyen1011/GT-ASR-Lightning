import os
import sys
import json
import argparse
from tqdm import tqdm

from tools import get_phn2attr_dict, word2phone, get_backend_separator

GSC_KWS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]

backend, separator = get_backend_separator('en')
gsc_phn = [word2phone(word, 'en', backend, separator) for word in GSC_KWS]

with open('data/mswc/metadata.json', 'r') as f:
    all_data = json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', type=str, default='en', required=True, help='language to preprocess')
    parser.add_argument('--attr', '-a', type=str, default='MP', help='attributes used to select keywords')
    parser.add_argument('--pv', '-p', action='store_true', help='whether to use place of vowels')
    parser.add_argument('--num', '-n', type=int, default=None, help='select keywords based on number of occurrences')
    parser.add_argument('--topk', '-t', type=int, default=None, help='select the top k keywords')
    parser.add_argument('--len', type=int, default=None, help='constraint to length of keywords')
    parser.add_argument('--ood', '-o', action='store_true', help='generate out of domain testing data')
    parser.add_argument('--gsc', '-g', action='store_true', help='use keywords from GSC')
    args = parser.parse_args()

    if args.pv:
        attr_path = args.attr
    else:
        attr_path = attr_path.replace('P', 'p')
    output_dir = f'data/{args.lang}-mswc-{attr_path}'
    if args.topk:
        output_dir += f'-top{args.topk}'
    elif args.num:
        output_dir += f'-thresh{args.num}'
    if args.gsc:
        output_dir += f'-gsc'

    if args.len:
        output_dir += f'-len{args.len}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    phn2attr = get_phn2attr_dict(attrs=args.attr, P_of_vowel=args.pv)
    backend, separator = get_backend_separator(args.lang)

    gsc_attr = set()
    for phn_seq in gsc_phn:
        for phn in phn_seq.split():
            gsc_attr.add(phn2attr[phn])
    print (gsc_attr)

    lang_data = all_data[args.lang]
    sorted_wordcounts = dict(sorted(lang_data["wordcounts"].items(), key=lambda x: x[1], reverse=True))

    word_clause_attr = {}
    word_clause_phn = {}
    word_clause_chr = {}
    total_counts = 0
    for word, count in tqdm(sorted_wordcounts.items()):
        if args.num and count < args.num or count == 0:
            break
        if args.topk and len(word_clause_attr) == args.topk:
            break
        if args.len and len(word) < args.len:
            continue
        if args.lang == 'en' and args.gsc and word not in GSC_KWS:
            continue

        chr_seq = ' '.join(list(word))
        phn_seq = word2phone(word, args.lang, backend, separator)
        jump = False
        for phn in phn_seq.split(): 
            if phn not in phn2attr:
                jump = True
                break
        if jump:
            continue
        attr_seq =  ' '.join(phn2attr[phn] for phn in phn_seq.split())
        for attr in attr_seq.split(): 
            if attr not in gsc_attr and args.gsc:
                jump = True
                break
        if jump:
            continue

        if attr_seq not in word_clause_attr:
            word_clause_attr[attr_seq] = word
            word_clause_phn[phn_seq] = word
            word_clause_chr[chr_seq] = word
            total_counts += count
        else:
            continue
    
    lexicon_attr = {word: units for units, word in word_clause_attr.items()}
    lexicon_phn = {word: units for units, word in word_clause_phn.items()}
    lexicon_chr = {word: units for units, word in word_clause_chr.items()}
    all_attr, all_phn, all_chr = {}, {}, {}
    i = 0
    for word, units in lexicon_attr.items():
        for unit in units.split():
            if unit not in all_attr:
                all_attr[unit] = i
                i += 1
    i = 0
    for word, units in lexicon_phn.items():
        for unit in units.split():
            if unit not in all_phn:
                all_phn[unit] = i
                i += 1
    i = 0
    for word, units in lexicon_chr.items():
        for unit in units.split():
            if unit not in all_chr:
                all_chr[unit] = i
                i += 1

    lang_data['wordcounts'] = {word: sorted_wordcounts[word] for word in lexicon_attr.keys()}
    lang_data['filenames'] = {word: lang_data['filenames'][word] for word in lexicon_attr.keys()}
    lang_data['number_of_words'] = len(lexicon_attr)
    lang_data['number_of_samples'] = total_counts
            
    with open(f'{output_dir}/metadata.json', 'w', encoding='utf8') as f:
        json.dump(lang_data, f, indent=4, ensure_ascii=False)
    
    with open(f'{output_dir}/vocab.txt', 'w') as f:
        for word, units in lexicon_attr.items():
            f.write(word)
            f.write('\n')
    
    with open(f'{output_dir}/lexicon_{attr_path}.txt', 'w', encoding='utf8') as f:
        for word, units in lexicon_attr.items():
            f.write(word)
            f.write(' ')
            f.write(units)
            f.write('\n')

    with open(f'{output_dir}/lexicon_phone.txt', 'w', encoding='utf8') as f:
        for word, units in lexicon_phn.items():
            f.write(word)
            f.write(' ')
            f.write(units)
            f.write('\n')
    
    with open(f'{output_dir}/lexicon_chr.txt', 'w', encoding='utf8') as f:
        for word, units in lexicon_chr.items():
            f.write(word)
            f.write(' ')
            f.write(units)
            f.write('\n')

    with open(f'{output_dir}/{attr_path}.json', 'w', encoding='utf8') as f:
        json.dump(all_attr, f, indent=4, ensure_ascii=False)
    
    with open(f'{output_dir}/phone.json', 'w', encoding='utf8') as f:
        json.dump(all_phn, f, indent=4, ensure_ascii=False)
    
    with open(f'{output_dir}/char.json', 'w', encoding='utf8') as f:
        json.dump(all_chr, f, indent=4, ensure_ascii=False)
    
    with open(f'{output_dir}/{attr_path}.txt', 'w', encoding='utf8') as f:
        for unit, idx in all_attr.items():
            f.write(unit)
            f.write('\n')
        f.write('<blank>\n')
        f.write('|')
    
    with open(f'{output_dir}/phone.txt', 'w', encoding='utf8') as f:
        for unit, idx in all_phn.items():
            f.write(unit)
            f.write('\n')
        f.write('<blank>\n')
        f.write('|')
    
    with open(f'{output_dir}/char.txt', 'w', encoding='utf8') as f:
        for unit, idx in all_chr.items():
            f.write(unit)
            f.write('\n')
        f.write('<blank>\n')
        f.write('|')
    
    with open(f'data/mswc/metadata_{args.lang}.json', 'r') as f:
        json.dump(all_data[args.lang], f, indent=4)