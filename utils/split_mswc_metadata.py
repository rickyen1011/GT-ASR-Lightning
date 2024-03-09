import os
import json
import pandas as pd
import argparse

root = '/mnt/disk2/rickyen/Datasets/mswc/splits'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', type=str, nargs='+', default=['en'], required=True, help='language to preprocess')
    parser.add_argument('--dir', '-d', type=str, required=True, help='directory that contains metadata to split')
    parser.add_argument('--num', '-n', type=int, default=None, help='number of samples per word for testing and development')
    parser.add_argument('--thresh', '-t', type=int, default=None, help='the most number of samples per word for testing and development')
    args = parser.parse_args()
    
    splits = []
    for lang in args.lang:
        splits.append(pd.read_csv(f'{root}/{lang}_splits.csv'))
    splits = pd.concat(splits)

    print (splits)

    with open(os.path.join(args.dir, 'metadata.json'), 'r') as f:
        all_data = json.load(f)
    
    
    for split in ['TRAIN']:
        metadata = {}
        metadata['languages'] = all_data['languages'].copy()
        metadata['dataset'] = 'MSWC'
        metadata['number_of_words'] = all_data['number_of_words'].copy()
        metadata['number_of_samples'] = {}
        metadata['wordcounts'] = {}
        metadata['filenames'] = {}
    
    
        for s, link in zip(splits['SET'].values.tolist(), splits['LINK'].values.tolist()):
            kw, path = link.split('/')
            lang = path.split('_')[2]
            if lang not in metadata['number_of_samples']:
                metadata['number_of_samples'][lang] = 0
            if kw not in all_data['wordcounts'] or s != split:
                continue
            if kw not in metadata['wordcounts']:
                metadata['wordcounts'][kw] = 0
                metadata['filenames'][kw] = []

            if path in all_data['filenames'][kw]:
                metadata['wordcounts'][kw] += 1
                metadata['number_of_samples'][lang] += 1
                filename = os.path.join(path.split('_')[2], 'clips', kw, path)
                metadata['filenames'][kw].append(filename)
        
        with open(f'{args.dir}/{split.lower()}.json', 'w', encoding='utf8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    for split in ['DEV', 'TEST']:
        metadata = {}
        metadata['languages'] = all_data['languages'].copy()
        metadata['dataset'] = 'MSWC'
        metadata['number_of_words'] = all_data['number_of_words'].copy()
        metadata['number_of_samples'] = {}
        metadata['wordcounts'] = {}
        metadata['filenames'] = {}
    
    
        for s, link in zip(splits['SET'].values.tolist(), splits['LINK'].values.tolist()):
            kw, path = link.split('/')
            lang = path.split('_')[2]
            if lang not in metadata['number_of_samples']:
                metadata['number_of_samples'][lang] = 0

            if kw not in all_data['wordcounts'] or s != split:
                continue
            if kw not in metadata['wordcounts']:
                metadata['wordcounts'][kw] = 0
                metadata['filenames'][kw] = []

            if args.thresh and metadata['wordcounts'][kw] == args.thresh:
                continue
            if path in all_data['filenames'][kw]:
                metadata['wordcounts'][kw] += 1
                metadata['number_of_samples'][lang] += 1
                filename = os.path.join(lang, 'clips', kw, path)
                metadata['filenames'][kw].append(filename)
        
        wordcounts = metadata['wordcounts'].copy()
        if args.num and count < args.num:
            for kw, count in wordcounts.items():
                lang = metadata['filenames'][kw][0].split('/')[0]
                del metadata['wordcounts'][kw]
                del metadata['filenames'][kw]
                metadata['number_of_samples'][lang] -= count
                metadata['number_of_words'][lang] -= 1

        with open(f'{args.dir}/{split.lower()}.json', 'w', encoding='utf8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)