import os
import json

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset

from utils.text_process import TextTransform

class Dataset(Dataset):
    def __init__(self, root_dir, data_dir, json_file, lexicon_file, split='train'):
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
        
        self.get_metadata(data_file)
    
    def get_metadata(self, json_file):
        with open(json_file, 'r') as f:
            all_data = json.load(f)
        for word, filenames in all_data['filenames'].items():
            seq = self.lexicon[word]
            for filename in filenames:
                if self.split:
                    path = f'{self.root_dir}/{self.split}/{word}/{filename}'
                else:
                    path = f'{self.root_dir}/{filename}'
                self.metadata.append((path, word, seq))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        transcript_dict = {}
        wav_path, word, transcript = self.metadata[idx]
        transcript_dict[self.attributes[-1]] = transcript
        for i, attr in enumerate(self.attributes[:-1]):
            attr_trans = ' '.join([attr.split('-')[i] for attr in transcript.split()])
            transcript_dict[attr] = attr_trans

        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        return waveform, word, transcript_dict

class Collate:
    def __init__(self, data_dir, token_files, attributes, sample_rate=16000, subsample_factor=0.02):
        self.text_transform = {}
        for attr, token_file in zip(attributes, token_files)
            token_file = os.path.join(data_dir, token_file)
            self.text_transform[attr] = TextTransform(token_file=token_file)

        self.sample_rate = sample_rate
        self.subsample_factor = subsample_factor

    def collate_fn(self, data):
        ''' Process batch examples '''
        waveforms = []
        labels_dict = {}
        references_dict = {}
        for attr in self.text_transform.keys():
            labels_dict[attr] = []
            references_dict[attr] = []
        references_word = []
        input_lengths = []
        label_lengths = []
        
        for (waveform, word, transcript_dict) in data:
            waveform = waveform.squeeze(0)
            waveforms.append(waveform)

            # Labels
            for attr, transcript in transcript_dict.items():
                references_dict[attr].append(transcript) # Actual Sentence
                label = torch.Tensor(self.text_transform[attr].text_to_int(transcript)) # Integer representation of sentence
                labels_dict[attr].append(label)
            references_word.append(word) # Actual Sentence

            # Lengths (time)
            input_lengths.append(waveform.shape[0] // (self.sample_rate * self.subsample_factor) -1) # account for subsampling of time dimension
            label_lengths.append(len(label))

        # Pad batch to length of longest sample
        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        for attr, labels in labels_dict:
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
            labels_dict[attr] = labels
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        return waveforms, labels_dict, input_lengths, label_lengths, references_dict, references_word