import re
import json

from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from phonemizer import phonemize

def get_phn2attr_dict(attrs=['M'], P_of_vowel=True):
    attr2phn_dict = {'M': {}, 'P': {}, 'V': {}, 'B': {}, 'R': {}}

    # Manner of articulation
    attr2phn_dict['M']['apr'] = ('w', 'ɹ', 'j', 'l', 'ʎ', 'ɰ', 'ɻ', 'ɫ', 'lʲ', 'ʋ', 'ɭ')
    attr2phn_dict['M']['flp'] = ('r', 'ʀ', 'ɾ', 'rʲ')
    attr2phn_dict['M']['frc'] = ('f', 'h', 'v', 'z', 's', 'ʃ', 'θ', 'ʒ', 'ð', 'x', 
                                 'ɕ', 'ɧ', 'ʂ', 'ɣ', 'ç', 'ʁ', 'ʑ', 'fʲ', 'vʲ', 'sʲ',
                                 'X', 'ɬ')
    attr2phn_dict['M']['afr'] = ('dʒ', 'tʃ', 'ts', 'dz', 'tɕ', 'ʈʂ', 'pf', 'tʃʲ')
    attr2phn_dict['M']['nas'] = ('m', 'n', 'ŋ', 'ɲ', 'mʲ', 'nʲ', 'N')
    attr2phn_dict['M']['stp'] = ('b', 'd', "p", 'ɡ', 't', 'k', 'ʔ', 'c', 'q', 'pʲ',
                                 'kʲ', 'bʲ', 'ɡʲ', 'd[', 't[', 'tʲ', 'dʲ', 'd̪', 't̪')
    attr2phn_dict['M']['vwl'] = ('æ', 'ɚ', 'ʌ', 'ɪ', 'ɛ', 'i', 'u', 'ɑ', 'ɔ', 'ʊ',
                                 'o', 'a', 'e', 'ə', 'œ', 'ɯ', 'ɤ', 'ɵ', 'ø', 'ɨ', 
                                 'ʏ', 'ʉ', 'ɐ', 'y', 'oʊ', 'eɪ', 'aɪ', 'ɔɪ', 'ᵻ', 'aʊ', 'ɜ',
                                 'œ̃', 'ɔ̃', 'ɛ̃', 'œy', 'ɛɪ', 'ʌʊ', 'eʊ', 'ɑ̃', 'ɒ')
    # Place of articulation
    attr2phn_dict['P']['blb'] = ('b', 'm', 'p', 'pf', 'mʲ', 'pʲ', 'bʲ')
    attr2phn_dict['P']['lbd'] = ('f', 'v', 'fʲ', 'vʲ', 'ʋ')
    attr2phn_dict['P']['dnt'] = ('ð', 'θ', 't̪', 'd̪')
    attr2phn_dict['P']['alv'] = ('d', 'l', 'n', 's', 't', 'z', 'ɹ', 'ts', 'dz', 'r', 'ɾ', 
                                 'ɫ', 'lʲ', 'sʲ', 'nʲ', 'rʲ', 'd[', 't[', 'tʲ', 'dʲ', 'ɬ')
    attr2phn_dict['P']['pla'] = ('tʃ', 'dʒ', 'ʃ', 'ʒ', 'ɧ', 'tʃʲ')
    attr2phn_dict['P']['rfx'] = ('ɻ', 'ʂ', 'ʈʂ', 'ɭ')
    attr2phn_dict['P']['pal'] = ('j', 'ɲ', 'ʎ', 'c', 'ɕ', 'tɕ', 'ç', 'ʑ')
    attr2phn_dict['P']['vel'] = ('ɡ', 'k', 'ŋ', 'w', 'x', 'ɣ', 'ɰ', 'kʲ', 'ɡʲ')
    attr2phn_dict['P']['uvl'] = ('q', 'ʀ', 'ʁ', 'N', 'X')
    attr2phn_dict['P']['glt'] = ('h', 'ʔ')

    if not P_of_vowel:
        attr2phn_dict['P']['vwl'] = ('æ', 'ɚ', 'ʌ', 'ɪ', 'ɛ', 'i', 'u', 'ɑ', 'ɔ', 'ʊ',
                                      'o', 'a', 'e', 'ə', 'œ', 'ɯ', 'ɤ', 'ɵ', 'ø', 'ɨ', 
                                      'ʏ', 'ʉ', 'ɐ', 'y', 'oʊ', 'eɪ', 'aɪ', 'ɔɪ', 'ᵻ', 'aʊ', 'ɜ',
                                      'œ̃', 'ɔ̃', 'ɛ̃', 'œy', 'ɛɪ', 'ʌʊ', 'eʊ', 'ɑ̃', 'ɒ')
    else:
        attr2phn_dict['P']['hgh'] = ('i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u')
        attr2phn_dict['P']['smh'] = ('ɪ', 'ʏ', 'ʊ')
        attr2phn_dict['P']['umd'] = ('e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o')
        attr2phn_dict['P']['mid'] = ('ə')
        attr2phn_dict['P']['lmd'] = ('ɛ', 'œ', 'ɜ', 'ʌ', 'ɔ',  'ɔ̃', 'œ̃', 'ɛ̃')
        attr2phn_dict['P']['sml'] = ('æ', 'ɐ')
        attr2phn_dict['P']['low'] = ('a', 'ɑ', 'ɒ', 'ɑ̃')
        attr2phn_dict['P']['unk'] = ('oʊ', 'eɪ', 'aɪ', 'œy', 'ɛɪ', 'ʌʊ', 'eʊ', 'ᵻ', 'aʊ', 'ɚ', 'ɔɪ')

    # Voiced or voiceless
    attr2phn_dict['V']['vcd'] = ('b', 'm', 'v', 'mʲ', 'vʲ', 'ʋ', 'bʲ', 'ð', 'd̪', 'n', 'd', 'dz',
                                 'ɣ', 'ɡ', 'z', 'ʒ', 'ɻ', 'dʒ', 'j', 'ʎ', 'ʁ', 'w', 'd[', 'ɰ',
                                 'N', 'ʀ', 'ɲ', 'ɾ', 'r', 'rʲ', 'ɫ', 'ʑ', 'nʲ', 'l', 'ŋ', 'ɹ',
                                 'lʲ', 'dʲ', 'ɡʲ', 'ɭ', 'æ', 'ɚ', 'ʌ', 'ɪ', 'ɛ', 'i', 'u', 'ɑ', 
                                 'ɔ', 'ʊ', 'o', 'a', 'e', 'ə', 'œ', 'ɯ', 'ɤ', 'ɵ', 'ø', 'ɨ', 
                                 'ʏ', 'ʉ', 'ɐ', 'y', 'oʊ', 'eɪ', 'aɪ', 'ɔɪ', 'ᵻ', 'aʊ', 'ɜ',
                                 'œ̃', 'ɔ̃', 'ɛ̃', 'œy', 'ɛɪ', 'ʌʊ', 'eʊ', 'ɑ̃', 'ɒ')
    attr2phn_dict['V']['vls'] = ('p', 'f', 'pf', 'fʲ', 'pʲ', 'θ', 't̪', 's', 't', 'x', 'k', 'c',
                                 'ʃ', 'ʈʂ', 'ʂ', 'tʃ', 'ts', 'ɕ', 't[', 'h', 'q', 'ʔ', 'X', 'ɧ',
                                 'ç', 'tʃʲ', 'sʲ', 'tʲ', 'kʲ', 'ɬ')

    # Backness of vowel
    attr2phn_dict['B']['fnt'] = ('i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'y', 'ʏ', 'ø', 'œ', 'œ̃', 'œy',
                                 'ɛ̃', 'ɛɪ', 'eɪ', 'aɪ')
    attr2phn_dict['B']['cnt'] = ('ɨ', 'ə', 'ɜ', 'ɐ', 'ʉ', 'ɵ', 'ɚ')
    attr2phn_dict['B']['bck'] = ('ɯ', 'ɤ', 'ʌ', 'ɑ', 'u', 'o', 'ʊ', 'ɔ', 'ɒ', 'oʊ', 'ɑ̃', 'ɔ̃',
                                 'ʌʊ')
    attr2phn_dict['B']['unk'] = ('eʊ', 'ɔɪ', 'ᵻ', 'aʊ')
    attr2phn_dict['B']['con'] = ('b', 'm', 'v', 'mʲ', 'vʲ', 'ʋ', 'bʲ', 'ð', 'd̪', 'n', 'd', 'dz',
                                 'ɣ', 'ɡ', 'z', 'ʒ', 'ɻ', 'dʒ', 'j', 'ʎ', 'ʁ', 'w', 'd[', 'ɰ',
                                 'N', 'ʀ', 'ɲ', 'ɾ', 'r', 'rʲ', 'ɫ', 'ʑ', 'nʲ', 'l', 'ŋ', 'ɹ',
                                 'lʲ', 'dʲ', 'ɡʲ', 'p', 'f', 'pf', 'fʲ', 'pʲ', 'θ', 't̪', 's', 't', 
                                 'x', 'k', 'c', 'ʃ', 'ʈʂ', 'ʂ', 'tʃ', 'ts', 'ɕ', 't[', 'h', 'q', 
                                 'ʔ', 'X', 'ɧ', 'ç', 'tʃʲ', 'sʲ', 'tʲ', 'kʲ', 'ɭ', 'ɬ')
    
    attr2phn_dict['R']['rnd'] = ('y', 'ʏ', 'ø', 'œ', 'ʉ', 'ɵ', 'u', 'o', 'ʊ', 'ɔ', 'ɒ', 'œy', 'oʊ')
    attr2phn_dict['R']['unr'] = ('i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ɨ', 'ə', 'ɜ', 'ɐ', 'ɚ', 'ɯ', 'ɤ', 'ʌ', 'ɑ', 'eɪ', 'ɛɪ', 'aɪ')
    attr2phn_dict['R']['unk'] = ('eʊ', 'ɔɪ', 'ᵻ', 'aʊ', 'ʌʊ')
    attr2phn_dict['R']['con'] = ('b', 'm', 'v', 'mʲ', 'vʲ', 'ʋ', 'bʲ', 'ð', 'd̪', 'n', 'd', 'dz',
                                 'ɣ', 'ɡ', 'z', 'ʒ', 'ɻ', 'dʒ', 'j', 'ʎ', 'ʁ', 'w', 'd[', 'ɰ',
                                 'N', 'ʀ', 'ɲ', 'ɾ', 'r', 'rʲ', 'ɫ', 'ʑ', 'nʲ', 'l', 'ŋ', 'ɹ',
                                 'lʲ', 'dʲ', 'ɡʲ', 'p', 'f', 'pf', 'fʲ', 'pʲ', 'θ', 't̪', 's', 't', 'x', 'k', 'c',
                                 'ʃ', 'ʈʂ', 'ʂ', 'tʃ', 'ts', 'ɕ', 't[', 'h', 'q', 'ʔ', 'X', 'ɧ',
                                 'ç', 'tʃʲ', 'sʲ', 'tʲ', 'kʲ', 'ɭ', 'ɬ')

    phn2attr = {}
    for attr in attrs:
        for attribute, phones in attr2phn_dict[attr].items():
            for phn in phones:
                if phn not in phn2attr:
                    phn2attr[phn] = []
                phn2attr[phn].append(attribute)

    for phn, attr in phn2attr.items():
        phn2attr[phn] = '-'.join(attr)

    return phn2attr

def get_all_phns():
    all_phns = set()
    # for lang in ['ky','ru','nl','tt','sv']:
    for lang in ['ky', 'nl', 'ru', 'sv', 'tt']:
        if lang == 'en':
            with open('data/librispeech-phone/phones.json', 'r') as f:
                d = json.load(f)
                for phn in d.keys():
                    all_phns.add(phn)
        else:
            with open(f'data/cv-{lang}-full-phone/phones.json', 'r') as f:
                d = json.load(f)
                for phn in d.keys():
                    if phn == 'oe':
                        print (lang)
                    all_phns.add(phn)
    
    print (len(all_phns))
    phn2M = get_phn2attr_dict(attr='M')
    phn2P = get_phn2attr_dict(attr='P')
    phn2voice = get_phn2attr_dict(attr='V')

    for phn in all_phns:
        if phn not in phn2voice:
            print (phn)

    return all_phns

def check_1to1(phns):
    attr2phn = {'M-P-V-B-R': {}}
    phn2manner = get_phn2attr_dict(attr='M')
    phn2place = get_phn2attr_dict(attr='P')
    phn2voicing = get_phn2attr_dict(attr='V')
    phn2backness = get_phn2attr_dict(attr='B')
    phn2round = get_phn2attr_dict(attr='R')

    for phn in phns:
        attr = '-'.join([phn2manner[phn],phn2place[phn],phn2voicing[phn],phn2backness[phn],phn2round[phn]])
        if attr in attr2phn['M-P-V-B-R']:
            print (attr+': '+phn+','+attr2phn['M-P-V-B-R'][attr])
        else:
            attr2phn['M-P-V-B-R'][attr] = phn

def get_backend_separator(lang):
    if lang == 'en':
        backend = EspeakBackend('en-us')
    elif lang == 'it':
        backend = EspeakBackend('it')
    elif lang == 'ru':
        backend = EspeakBackend('ru')
    elif lang == 'sv':
        backend = EspeakBackend('sv')
    elif lang == 'ky':
        backend = EspeakBackend('ky')
    elif lang == 'nl':
        backend = EspeakBackend('nl')
    elif lang == 'fr':
        backend = EspeakBackend('fr-fr')
    elif lang == 'es':
        backend = EspeakBackend('es')
    elif lang == 'tt':
        backend = EspeakBackend('tt')
    else:
        raise NotImplementedError
    
    separator = Separator(phone=' ', word=None)

    return backend, separator

def word2phone(word, lang, backend, separator):
    phones = backend.phonemize([word], separator=separator, strip=True)[0]
    phones = re.sub('ː', '', phones)

    if lang == 'en': 
        phones = re.sub('ɚ',' ɚ', phones)
        phones = re.sub('ɹ',' ɹ', phones)
        phones = re.sub('ə', ' ə', phones)
        phones = re.sub('əl', 'ə l', phones)
        phones = re.sub('ju', 'j u', phones)
        phones = re.sub('ja', 'j a', phones)
        phones = re.sub('̩', '', phones)
        phones = re.sub('aɪɚ', 'a ɪ ɚ', phones)

    elif lang == 'it':
        phones = re.sub('ss', 's', phones)
    elif lang == 'sv':
        phones = re.sub('sx', 's x', phones)
    elif lang == 'ru':
        phones = re.sub('\"', '', phones)
        phones = re.sub("\^", "", phones)
        phones = re.sub("ja", 'j a', phones)
        phones = re.sub("ju", 'j u', phones)
        phones = re.sub("ʲ", '', phones)
    elif lang == 'ky':
        phones = re.sub(':', '', phones)
        phones = re.sub('Z', 'z', phones)
        phones = re.sub('S', 's', phones)
        phones = re.sub('N', 'n', phones)
        phones = re.sub('X', 'x', phones)
        phones = re.sub('oe', 'o e', phones)
    elif lang == 'fr':
        phones = re.sub('əl', 'ə l', phones)
        phones = re.sub('ea', 'e a', phones)
        phones = re.sub('ʊə', 'ʊ ə', phones)
        phones = re.sub('iə', 'i ə', phones)
        phones = re.sub('əʊ', 'ə ʊ', phones)
        phones = re.sub('eə', 'e ə', phones)
        phones = re.sub('aɪə', 'aɪ ə', phones)
    elif lang == 'nl':
        phones = re.sub('yʊ', 'y ʊ', phones)
        phones = re.sub('eʊ', 'e ʊ', phones)
        # phones = re.sub('œy', 'œ y', phones)
        # phones = re.sub('ɛɪ', 'ɛ ɪ', phones)
        phones = re.sub("ʲ", '', phones)
        phones = re.sub("ɾ", 'r', phones)
    elif lang == 'tt':
        phones = re.sub('ɫ', 'l', phones)

    phones = re.sub(' +', ' ', phones)
    for l in ['en', 'nl', 'fr', 'it', 'tt', 'ky', 'ta']:
        phones = phones.replace(f'({l})', '')
    return phones

if __name__ == '__main__':
    all_phns = get_all_phns()
    print (len(all_phns))
    # check_1to1(all_phns)

    all_phns_dict = {phn: i for i, phn in enumerate(list(all_phns))}
    multipath = 'cv-ky_nl_ru_sv_tt-full-phone'
    with open(f'data/{multipath}/phones.json', 'w', encoding='utf8') as f:
        json.dump(all_phns_dict, f, indent=4, ensure_ascii=False)