import os
import json

class TextTransform:
    ''' Map units to integers and vice versa '''
    def __init__(self, token_file):
        with open(token_file, 'r') as f:
            self.unit_map = json.load(f)
        self.unit_map["<blank>"] = len(self.unit_map)
        self.index_map = {} 
        for unit, i in self.unit_map.items():
            self.index_map[i] = unit

    def text_to_int(self, text):
        ''' Map text string to an integer sequence '''
        int_sequence = []
        for u in text.split():
            idx = self.unit_map[u]
            int_sequence.append(idx)
        return int_sequence

    def int_to_text(self, labels):
        ''' Map integer sequence to text string '''
        string = []
        for i in labels:
            if i == len(self.unit_map) - 1: # blank char
                continue
            else:
                string.append(self.index_map[i])
        return ' '.join(string)