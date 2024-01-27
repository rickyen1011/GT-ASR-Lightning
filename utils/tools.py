import re
import json

from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from phonemizer import phonemize

def get_phn2attr_dict(attrs=['M'], P_of_vowel=True):
    attr2phn_dict = {'M': {}, 'P': {}, 'V': {}, 'B': {}, 'R': {}}

    # Manner of articulation
    attr2phn_dict['M']['apr'] = ('w', 'ɹ', 'j', 'l', 'ʎ', 'ɰ', 'ɻ', 'ɫ', 'lʲ', 'ʋ', 'ɭ', 'l̩', 'ɭʲ')
    attr2phn_dict['M']['flp'] = ('r', 'ʀ', 'ɾ', 'rʲ', 'r̝̊', 'r̝', 'r̩')
    attr2phn_dict['M']['frc'] = ('f', 'h', 'v', 'z', 's', 'ʃ', 'θ', 'ʒ', 'ð', 'x', 
                                 'ɕ', 'ɧ', 'ʂ', 'ɣ', 'ç', 'ʁ', 'ʑ', 'fʲ', 'vʲ', 'sʲ',
                                 'X', 'ɬ', 'ç', 'β', 'ʂʲ', 'ʒʲ', 'ɕʲ', 'xʲ', 's^')
    attr2phn_dict['M']['afr'] = ('dʒ', 'tʃ', 'ts', 'dz', 'tɕ', 'ʈʂ', 'pf', 'tʃʲ', 'tɕʲ', 'dʑʲ', 'tsʲ',
                                 'dʑ', 'tʃʰ', 'tsʰ', 't͡s', 't͡ʃ')
    attr2phn_dict['M']['nas'] = ('m', 'n', 'ŋ', 'ɲ', 'mʲ', 'nʲ', 'N', 'ɳ', 'ɲʲ', 'nʲʲ')
    attr2phn_dict['M']['stp'] = ('b', 'd', "p", 'ɡ', 't', 'k', 'ʔ', 'c', 'q', 'pʲ', 'bʰ', 'ɡʰ', 'kʰ', 'tʰ',
                                 'kʲ', 'bʲ', 'ɡʲ', 'd[', 't[', 'tʲ', 'dʲ', 'd̪', 't̪', 'ɖ', 'ʈ', 'ɟ', 'dˤ', 'pʰ')
    attr2phn_dict['M']['vwl'] = ('æ', 'ɚ', 'ʌ', 'ɪ', 'ɛ', 'i', 'u', 'ɑ', 'ɔ', 'ʊ',
                                 'o', 'a', 'e', 'ə', 'œ', 'ɯ', 'ɤ', 'ɵ', 'ø', 'ɨ', 'ɐ̃', 
                                 'ʏ', 'ʉ', 'ɐ', 'y', 'oʊ', 'eɪ', 'aɪ', 'ɔɪ', 'ᵻ', 'aʊ', 'ɜ',
                                 'œ̃', 'ɔ̃', 'ɛ̃', 'œy', 'ɛɪ', 'ʌʊ', 'eʊ', 'ɑ̃', 'ɒ', 'ũ', 'i̯', 'ai', 'au')
    # Place of articulation
    attr2phn_dict['P']['blb'] = ('b', 'm', 'p', 'pf', 'mʲ', 'pʲ', 'bʲ', 'β', 'bʰ', 'pʰ')
    attr2phn_dict['P']['lbd'] = ('f', 'v', 'fʲ', 'vʲ', 'ʋ')
    attr2phn_dict['P']['dnt'] = ('ð', 'θ', 't̪', 'd̪')
    attr2phn_dict['P']['alv'] = ('d', 'l', 'n', 's', 't', 'z', 'ɹ', 'ts', 'dz', 'r', 'ɾ', 'r̝̊', 'r̝', 'r̩', 'dˤ', 't͡s', 's^',
                                 'ɫ', 'lʲ', 'sʲ', 'nʲ', 'rʲ', 'd[', 't[', 'tʲ', 'dʲ', 'ɬ', 'l̩', 'tsʲ', 'nʲʲ', 'tʰ', 'tsʰ')
    attr2phn_dict['P']['pla'] = ('tʃ', 'dʒ', 'ʃ', 'ʒ', 'ɧ', 'tʃʲ', 'ʒʲ', 'tʃʰ', 't͡ʃ')
    attr2phn_dict['P']['rfx'] = ('ɻ', 'ʂ', 'ʈʂ', 'ɭ', 'ɖ', 'ʈ', 'ɳ', 'ʂʲ', 'ɭʲ')
    attr2phn_dict['P']['pal'] = ('j', 'ɲ', 'ʎ', 'c', 'ɕ', 'tɕ', 'ç', 'ʑ', 'ç', 'tɕʲ', 'dʑʲ', 'ɕʲ', 'dʑ',
                                 'ɲʲ', 'ɟ')
    attr2phn_dict['P']['vel'] = ('ɡ', 'k', 'ŋ', 'w', 'x', 'ɣ', 'ɰ', 'kʲ', 'ɡʲ', 'xʲ', 'ɡʰ', 'kʰ')
    attr2phn_dict['P']['uvl'] = ('q', 'ʀ', 'ʁ', 'N', 'X')
    attr2phn_dict['P']['glt'] = ('h', 'ʔ')

    if not P_of_vowel:
        attr2phn_dict['P']['vwl'] = ('æ', 'ɚ', 'ʌ', 'ɪ', 'ɛ', 'i', 'u', 'ɑ', 'ɔ', 'ʊ',
                                      'o', 'a', 'e', 'ə', 'œ', 'ɯ', 'ɤ', 'ɵ', 'ø', 'ɨ', 'ũ', 
                                      'ʏ', 'ʉ', 'ɐ', 'y', 'oʊ', 'eɪ', 'aɪ', 'ɔɪ', 'ᵻ', 'aʊ', 'ɜ',
                                      'œ̃', 'ɔ̃', 'ɛ̃', 'œy', 'ɛɪ', 'ʌʊ', 'eʊ', 'ɑ̃', 'ɒ', 'ɐ̃')
    else:
        attr2phn_dict['P']['hgh'] = ('i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u', 'ũ', 'i̯')
        attr2phn_dict['P']['smh'] = ('ɪ', 'ʏ', 'ʊ')
        attr2phn_dict['P']['umd'] = ('e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o')
        attr2phn_dict['P']['mid'] = ('ə')
        attr2phn_dict['P']['lmd'] = ('ɛ', 'œ', 'ɜ', 'ʌ', 'ɔ',  'ɔ̃', 'œ̃', 'ɛ̃')
        attr2phn_dict['P']['sml'] = ('æ', 'ɐ', 'ɐ̃')
        attr2phn_dict['P']['low'] = ('a', 'ɑ', 'ɒ', 'ɑ̃')
        attr2phn_dict['P']['unk'] = ('oʊ', 'eɪ', 'aɪ', 'œy', 'ɛɪ', 'ʌʊ', 'eʊ', 'ᵻ', 'aʊ', 'ɚ', 'ɔɪ', 'ai', 'au')

    # Voiced or voiceless
    attr2phn_dict['V']['vcd'] = ('b', 'm', 'v', 'mʲ', 'vʲ', 'ʋ', 'bʲ', 'ð', 'd̪', 'n', 'd', 'dz',
                                 'ɣ', 'ɡ', 'z', 'ʒ', 'ɻ', 'dʒ', 'j', 'ʎ', 'ʁ', 'w', 'd[', 'ɰ',
                                 'N', 'ʀ', 'ɲ', 'ɾ', 'r', 'rʲ', 'ɫ', 'ʑ', 'nʲ', 'l', 'ŋ', 'ɹ',
                                 'lʲ', 'dʲ', 'ɡʲ', 'ɭ', 'æ', 'ɚ', 'ʌ', 'ɪ', 'ɛ', 'i', 'u', 'ɑ', 
                                 'ɔ', 'ʊ', 'o', 'a', 'e', 'ə', 'œ', 'ɯ', 'ɤ', 'ɵ', 'ø', 'ɨ', 
                                 'ʏ', 'ʉ', 'ɐ', 'y', 'oʊ', 'eɪ', 'aɪ', 'ɔɪ', 'ᵻ', 'aʊ', 'ɜ',
                                 'œ̃', 'ɔ̃', 'ɛ̃', 'œy', 'ɛɪ', 'ʌʊ', 'eʊ', 'ɑ̃', 'ɒ', 'β', 'ɖ', 'ɳ', 'dʑʲ',
                                 'dʑ', 'ʒʲ', 'ɲʲ', 'ɟ')
    attr2phn_dict['V']['vls'] = ('p', 'f', 'pf', 'fʲ', 'pʲ', 'θ', 't̪', 's', 't', 'x', 'k', 'c',
                                 'ʃ', 'ʈʂ', 'ʂ', 'tʃ', 'ts', 'ɕ', 't[', 'h', 'q', 'ʔ', 'X', 'ɧ',
                                 'ç', 'tʃʲ', 'sʲ', 'tʲ', 'kʲ', 'ɬ', 'ç', 'ʈ', ' tɕʲ', 'tsʲ', 'ʂʲ', 
                                 'ɕʲ')

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
                                 'ʔ', 'X', 'ɧ', 'ç', 'tʃʲ', 'sʲ', 'tʲ', 'kʲ', 'ɭ', 'ɬ', 'ç', 'β')
    
    attr2phn_dict['R']['rnd'] = ('y', 'ʏ', 'ø', 'œ', 'ʉ', 'ɵ', 'u', 'o', 'ʊ', 'ɔ', 'ɒ', 'œy', 'oʊ')
    attr2phn_dict['R']['unr'] = ('i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ɨ', 'ə', 'ɜ', 'ɐ', 'ɚ', 'ɯ', 'ɤ', 'ʌ', 'ɑ', 'eɪ', 'ɛɪ', 'aɪ')
    attr2phn_dict['R']['unk'] = ('eʊ', 'ɔɪ', 'ᵻ', 'aʊ', 'ʌʊ')
    attr2phn_dict['R']['con'] = ('b', 'm', 'v', 'mʲ', 'vʲ', 'ʋ', 'bʲ', 'ð', 'd̪', 'n', 'd', 'dz',
                                 'ɣ', 'ɡ', 'z', 'ʒ', 'ɻ', 'dʒ', 'j', 'ʎ', 'ʁ', 'w', 'd[', 'ɰ',
                                 'N', 'ʀ', 'ɲ', 'ɾ', 'r', 'rʲ', 'ɫ', 'ʑ', 'nʲ', 'l', 'ŋ', 'ɹ',
                                 'lʲ', 'dʲ', 'ɡʲ', 'p', 'f', 'pf', 'fʲ', 'pʲ', 'θ', 't̪', 's', 't', 'x', 'k', 'c',
                                 'ʃ', 'ʈʂ', 'ʂ', 'tʃ', 'ts', 'ɕ', 't[', 'h', 'q', 'ʔ', 'X', 'ɧ',
                                 'ç', 'tʃʲ', 'sʲ', 'tʲ', 'kʲ', 'ɭ', 'ɬ', 'ç', 'β')

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
    elif lang == 'sv-SE':
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
    elif lang == 'es':
        backend = EspeakBackend('es')
    elif lang == 'de':
        backend = EspeakBackend('de')
    elif lang == 'ta':
        backend = EspeakBackend('ta')
    elif lang == 'lt':
        backend = EspeakBackend('lt')
    elif lang == 'ar':
        backend = EspeakBackend('ar')
    elif lang == 'fa':
        backend = EspeakBackend('fa')
    elif lang == 'pl':
        backend = EspeakBackend('pl')
    elif lang == 'tr':
        backend = EspeakBackend('tr')
    elif lang == 'cs':
        backend = EspeakBackend('cs')
    elif lang == 'sl':
        backend = EspeakBackend('sl')
    elif lang == 'lv':
        backend = EspeakBackend('lv')
    elif lang == 'vi':
        backend = EspeakBackend('vi')
    elif lang == 'ro':
        backend = EspeakBackend('ro')
    elif lang == 'el':
        backend = EspeakBackend('el')
    elif lang == 'sk':
        backend = EspeakBackend('sk')
    elif lang == 'pt':
        backend = EspeakBackend('pt')
    elif lang == 'as':
        backend = EspeakBackend('as')
    elif lang == 'ka':
        backend = EspeakBackend('ka')
    elif lang == 'mt':
        backend = EspeakBackend('mt')
    elif lang == 'gn':
        backend = EspeakBackend('gn')
    elif lang == 'cu':
        backend = EspeakBackend('cu')
    elif lang == 'or':
        backend = EspeakBackend('or')
    elif lang == 'et':
        backend = EspeakBackend('et')
    elif lang == 'eo':
        backend = EspeakBackend('eo')
    elif lang == 'ia':
        backend = EspeakBackend('ia')
    elif lang == 'id':
        backend = EspeakBackend('id')
    elif lang == 'ga-IE':
        backend = EspeakBackend('ga')
    elif lang == 'ca':
        backend = EspeakBackend('ca')
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
    elif lang == 'tt':
        phones = re.sub('ɫ', 'l', phones)
    elif lang == 'de':
        phones = re.sub('ɔø', 'ɔ ø', phones)
        phones = re.sub('ɔø', 'ɔ ø', phones)
    elif lang == 'ga-IE':
        phones = re.sub('A', 'a', phones)

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