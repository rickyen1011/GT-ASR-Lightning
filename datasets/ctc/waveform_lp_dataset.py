import os
import json
import random

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset

from utils.text_process import TextTransform

class Dataset(Dataset):
    def __init__(self, root_dir, data_dir, json_file, lexicon_file, split=None, portion=None):
        self.root_dir = root_dir
        self.split = split
        data_file = os.path.join(data_dir, json_file)
        self.metadata = []

        lexicon_file = os.path.join(data_dir, lexicon_file)
        self.lexicon = {}
        with open(lexicon_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                word, seq = line[0], ' '.join(line[1:])
                self.lexicon[word] = seq
        
        self.portion = portion
        self.get_metadata(data_file)
    
    def get_metadata(self, json_file):
        with open(json_file, 'r') as f:
            all_data = json.load(f)
            total_sample = list(all_data["number_of_samples"].values())[0]
        for word, filenames in all_data['filenames'].items():
            seq = self.lexicon[word]
            for filename in filenames:
                lang = filename.split('/')[0]
                if self.split and lang != self.split:
                    continue
                else:
                    path = f'{self.root_dir}/{filename}'
                if self.portion:
                    if len(self.metadata) < self.portion and random.random() < self.portion / total_sample:
                        self.metadata.append((path, lang, word, seq))
                    else:
                        continue
                else:
                    self.metadata.append((path, lang, word, seq))
        
        print (len(self.metadata))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_path, lang, word, transcript = self.metadata[idx]
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        return waveform, lang, word, transcript

class Collate:
    def __init__(self, data_dir, token_file, sample_rate=16000, subsample_factor=0.02):
        token_file = os.path.join(data_dir, token_file)
        self.text_transform = TextTransform(token_file=token_file)

        self.sample_rate = sample_rate
        self.subsample_factor = subsample_factor

    def collate_fn(self, data):
        ''' Process batch examples '''
        waveforms = []
        labels = []
        references = []
        references_word = []
        input_lengths = []
        label_lengths = []
        langs = []
        
        for (waveform, lang, word, transcript) in data:
            waveform = waveform.squeeze(0)
            waveforms.append(waveform)

            # Labels 
            references.append(transcript) # Actual Sentence
            references_word.append(word) # Actual Sentence
            label = torch.Tensor(self.text_transform.text_to_int(transcript)) # Integer representation of sentence
            labels.append(label)
            langs.append(lang)

            # Lengths (time)
            input_lengths.append(waveform.shape[0] // (self.sample_rate * self.subsample_factor) -1) # account for subsampling of time dimension
            label_lengths.append(len(label))

        # Pad batch to length of longest sample
        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        return waveforms, labels, input_lengths, label_lengths, references, references_word, langs