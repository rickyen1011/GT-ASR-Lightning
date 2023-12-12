import os
import sys
import json
import argparse
from tqdm import tqdm

from tools import get_phn2attr_dict, word2phone, get_backend_separator

backend, separator = get_backend_separator('en')

with open('data/mswc/metadata.json', 'r') as f:
    all_data = json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', type=str, nargs='+', default=['en'], required=True, help='language to preprocess')
    parser.add_argument('--attr', '-a', type=str, default='MP', help='attributes used to select keywords')
    parser.add_argument('--pv', '-p', action='store_true', help='whether to use place of vowels')
    parser.add_argument('--thresh', type=int, default=None, help='select keywords over a certain number of occurrences')
    parser.add_argument('--topk', '-t', type=int, default=None, help='select the top k keywords')
    parser.add_argument('--num', '-n', type=int, default=None, help='limit the number of utterances per keyword')
    parser.add_argument('--len', type=int, default=None, help='constraint to length of keywords')
    parser.add_argument('--ood', '-o', action='store_true', help='generate out of domain testing data')
    args = parser.parse_args()

    if args.pv:
        attr_path = args.attr
    else:
        attr_path = args.attr.replace('P', 'p')
    lang_path = '-'.join(args.lang)
    output_dir = f'data/mswc-{lang_path}-{attr_path}'
    if args.topk:
        output_dir += f'-top{args.topk}'
    elif args.thresh:
        output_dir += f'-thresh{args.thresh}'
    if args.num:
        output_dir += f'-num{args.num}'

    if args.len:
        output_dir += f'-len{args.len}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    phn2attr = get_phn2attr_dict(attrs=args.attr, P_of_vowel=args.pv)

    extract_data = {'languages': [], 'dataset': 'MSWC', 'number_of_words': {}, 'number_of_samples': {}, 'wordcounts': {}, 'filenames': {}}
    word_clause_attr = {}
    word_clause_phn = {}
    word_clause_chr = {}
    word_counts = {}
    word2files = {}
    total_counts = {}
    lang2nwords = {}

    for lang in args.lang:
        
        backend, separator = get_backend_separator(lang)
        lang_data = dict(all_data[lang])
        extract_data['languages'].append(lang_data['language'])
        sorted_wordcounts = dict(sorted(lang_data["wordcounts"].items(), key=lambda x: x[1], reverse=True))
        lang_data['wordcounts'] = sorted_wordcounts
        if lang not in total_counts:
            total_counts[lang] = 0
            lang2nwords[lang] = 0
        if not os.path.exists(f'data/mswc/metadata_{lang}.json'):
            with open(f'data/mswc/metadata_{lang}.json', 'w', encoding='utf8') as f:
                json.dump(lang_data, f, indent=4, ensure_ascii=False)

        for word, count in tqdm(sorted_wordcounts.items()):
            if args.thresh and count < args.thresh or count == 0:
                continue
            if args.topk and len(word_clause_attr) == args.topk:
                break
            if args.len and len(word) < args.len:
                continue
            if args.num and count > args.num:
                count = args.num
            if word in word_counts:
                continue

            chr_seq = ' '.join(list(word))
            phn_seq = word2phone(word, lang, backend, separator)
            
            jump = False
            for phn in phn_seq.split(): 
                if phn not in phn2attr:
                    print (lang, ' ', phn)
                    jump = True
                    break
            if jump:
                continue
            attr_seq =  ' '.join(phn2attr[phn] for phn in phn_seq.split())
            
            if attr_seq not in word_clause_attr:
                word_clause_attr[attr_seq] = word
                word_clause_phn[phn_seq] = word
                word_clause_chr[chr_seq] = word
                word_counts[word] = count
                word2files[word] = lang_data['filenames'][word][:count]
                lang2nwords[lang] += 1
                total_counts[lang] += count
            else:
                continue
    
    extract_data['wordcounts'] = word_counts
    extract_data['filenames'] = word2files
    extract_data['number_of_words'] = lang2nwords
    extract_data['number_of_samples'] = total_counts
        
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

                
    with open(f'{output_dir}/metadata.json', 'w', encoding='utf8') as f:
        json.dump(extract_data, f, indent=4, ensure_ascii=False)
        
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