import os
import json

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

from utils.text_process import TextTransform

class Dataset(Dataset):
    def __init__(self, root_dir, data_dir, json_file, lexicon_file, split=None):
        self.root_dir = root_dir
        self.split = split
        data_file = os.path.join(data_dir, json_file)
        self.metadata = {}

        lexicon_file = os.path.join(data_dir, lexicon_file)
        self.lexicon = {}
        with open(lexicon_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                word, seq = line[0], ' '.join(line[1:])
                self.lexicon[word] = seq
        
        self.dev = 'dev' in json_file

        self.languages = []
        self.get_metadata(data_file)
        self.num_data = sum([len(self.metadata[lang]) for lang in self.languages])
    
    def get_metadata(self, json_file):
        with open(json_file, 'r') as f:
            all_data = json.load(f)
        for word, filenames in all_data['filenames'].items():
            seq = self.lexicon[word]
            i = 0
            for filename in filenames:
                lang = filename.split('/')[0]
                if self.dev and lang in self.metadata and len(self.metadata[lang]) == 10000:
                    break
                if self.split and lang != self.split:
                    continue
                else:
                    path = f'{self.root_dir}/{filename}'
                    i += 1
                if lang not in self.metadata:
                    self.metadata[lang] = []
                    self.languages.append(lang)
                self.metadata[lang].append((path, word, seq))

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        language_idx = idx % len(self.languages)
        lang = self.languages[language_idx]
        sample_idx = idx // len(self.languages)
        wav_path, word, transcript = self.metadata[lang][sample_idx % len(self.metadata[lang])]
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        return waveform, word, transcript, lang

class Collate:
    def __init__(self, data_dir, token_file, langs_id=None, langs=None, sample_rate=16000, subsample_factor=0.02, spec_aug=False):
        token_file = os.path.join(data_dir, token_file)
        self.text_transform = TextTransform(token_file=token_file)

        self.langs = langs
        self.langs_id = langs_id
        self.sample_rate = sample_rate
        self.subsample_factor = subsample_factor
        self.spec_aug = spec_aug

    def collate_fn(self, data):
        ''' Process batch examples '''
        waveforms = []
        labels = []
        references = []
        references_word = []
        input_lengths = []
        label_lengths = []
        labels_d = []
        langs = []
        
        for (waveform, word, transcript, lang) in data:
            if self.spec_aug:
                # Convert waveform to spectrogram
                spectrogram_transform = T.Spectrogram(n_fft=400, win_length=400, hop_length=100, power=None)
                complex_spectrogram = spectrogram_transform(waveform)

                # Separate magnitude and phase
                magnitude = torch.abs(complex_spectrogram)
                phase = torch.angle(complex_spectrogram)

                # Apply SpecAugment
                # Frequency Masking: mask up to 30 frequency channels
                frequency_masking = T.FrequencyMasking(freq_mask_param=30)
                magnitude_masked = frequency_masking(magnitude)

                # Time Masking: mask up to 100 time steps
                time_masking = T.TimeMasking(time_mask_param=50)
                magnitude_masked = time_masking(magnitude_masked)

                spectrogram_augmented = torch.polar(magnitude_masked, phase)
                ispec = T.InverseSpectrogram(n_fft=400, win_length=400, hop_length=100)
                waveform = ispec(spectrogram_augmented)

            waveform = waveform.squeeze(0)
            waveforms.append(waveform)
            
            # Labels 
            references.append(transcript) # Actual Sentence
            references_word.append(word) # Actual Sentence
            label = torch.Tensor(self.text_transform.text_to_int(transcript)) # Integer representation of sentence
            labels.append(label)
            if self.langs:
                labels_d.append(self.langs.index(lang))
            else:
                assert self.langs_id != None
                labels_d.append(self.langs_id)
            langs.append(lang)

            # Lengths (time)
            input_lengths.append(waveform.shape[0] // (self.sample_rate * self.subsample_factor) -1) # account for subsampling of time dimension
            label_lengths.append(len(label))

        # Pad batch to length of longest sample
        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)
        labels_d = torch.tensor(labels_d, dtype=torch.long)

        # calc_nlangs(langs)
        return waveforms, labels, input_lengths, label_lengths, references, references_word, labels_d, langs

def calc_nlangs(langs):
    lang2count = {}
    for lang in langs:
        if lang not in lang2count:
            lang2count[lang] = 0
        lang2count[lang] += 1
    print (lang2count)