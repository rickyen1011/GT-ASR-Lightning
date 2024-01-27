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


class LPTextTransform(TextTransform):
    def __init__(self, token_file, target_token):
        with open(token_file, 'r') as f:
            self.unit_map = json.load(f)
        with open(target_token, 'r') as f:
            self.target_token = json.load(f)
        self.unit_map["<blank>"] = len(self.unit_map)
        for unit in self.target_token:
            if unit not in self.unit_map:
                u = random.choice(list(self.unit_map.keys()))
                while u in self.target_token:
                    u = random.choice(list(self.unit_map.keys()))
                self.unit_map[unit] = self.unit_map[u]
                del self.unit_map[u]
        self.index_map = {} 
        for unit, i in self.unit_map.items():
            self.index_map[i] = unit
        
        print (self.unit_map)
        print (self.index_map)