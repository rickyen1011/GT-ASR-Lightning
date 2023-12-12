import os
import json
import pandas as pd
import argparse

root = '/mnt/disk2/rickyen/Datasets/mswc/splits'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', type=str, nargs='+', default=['en'], required=True, help='language to preprocess')
    parser.add_argument('--dir', '-d', type=str, required=True, help='directory that contains metadata to split')
    parser.add_argument('--num', '-n', type=int, help='number samples per word for testing and development')
    args = parser.parse_args()
    
    splits = []
    for lang in args.lang:
        splits.append(pd.read_csv(f'{root}/{lang}_splits.csv'))
    splits = pd.concat(splits)

    print (splits)

    with open(os.path.join(args.dir, 'metadata.json'), 'r') as f:
        all_data = json.load(f)
    
    
    for split in ['TRAIN', 'DEV', 'TEST']:
        metadata = {}
        metadata['languages'] = all_data['languages']
        metadata['dataset'] = 'MSWC'
        metadata['number_of_words'] = all_data['number_of_words']
        metadata['number_of_samples'] = 0
        metadata['wordcounts'] = {}
        metadata['filenames'] = {}
    
    
        for s, link in zip(splits['SET'].values.tolist(), splits['LINK'].values.tolist()):
            kw, path = link.split('/')
            if kw not in all_data['wordcounts'] or s != split:
                continue
            if kw not in metadata['wordcounts']:
                metadata['wordcounts'][kw] = 0
            if kw not in metadata['filenames']:
                metadata['filenames'][kw] = []

            if split != 'TRAIN' and metadata['wordcounts'][kw] == args.num:
                continue
            if path in all_data['filenames'][kw]:
                metadata['wordcounts'][kw] += 1
                metadata['number_of_samples'] += 1
                filename = os.path.join(path.split('_')[2], 'clips', kw, path)
                metadata['filenames'][kw].append(filename)
        
        with open(f'{args.dir}/{split.lower()}.json', 'w', encoding='utf8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)