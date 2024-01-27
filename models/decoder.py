import os

import torch
import torch.nn as nn

from torchaudio.models.decoder import ctc_decoder


class GreedyDecoder(nn.Module):
    ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
    def __init__(self):
        super(GreedyDecoder, self).__init__()

    def forward(self, x):
        indices = torch.argmax(x, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        return indices.tolist()

def get_beam_decoder(
    split,
    test_config,
    model_config,
    nbest=5, 
    beam_size=50, 
    lm_weight=2, 
    word_score=0,
    blank_token='<blank>'
):

    lexicon_path = os.path.join(
        test_config[f"{split}_dataset"]["args"]["data_dir"], 
        test_config[f"{split}_dataset"]["args"]["lexicon_file"]
    )

    token_path = os.path.join(
        model_config[f"validation_dataset"]["args"]["data_dir"], 
        model_config[f"validation_dataset"]["beam_search_decoder"]["token_file"]
    )
        
    lm_path = os.path.join(
        test_config[f"{split}_dataset"]["args"]["data_dir"], 
        test_config[f"{split}_dataset"]["beam_search_decoder"]["lm_file"]
    )

    beam_search_decoder = ctc_decoder(
        lexicon=lexicon_path,
        tokens=token_path,
        lm=lm_path,
        nbest=nbest,
        beam_size=beam_size,
        lm_weight=lm_weight,
        word_score=word_score,
        blank_token=blank_token,
    )
    return beam_search_decoder