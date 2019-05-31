import re
import os
import bs4
import string
import pickle
import numpy as np
from lib.utils import create_path

def read_tmx(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.readlines()

    txt_str = ''
    for row in txt:
        txt_str += row

    txt_xml_soup = bs4.BeautifulSoup(txt_str, 'xml')
    en_txt = txt_xml_soup.findAll('tuv', {'xml:lang':"en"})
    en_txt = [element.text for element in en_txt]
    ch_txt = txt_xml_soup.findAll('tuv', {'xml:lang':"zh"})
    ch_txt = [element.text for element in ch_txt]
    
    return en_txt, ch_txt

def label_start_end(sentence):
    return '<START> ' + sentence + ' <END>'

def add_space_before_punct(sentence):
    return re.sub(r'([!"#$%&\'\(\)*+,-./:;<=>?@\[\\\]^_`{|}~–…])', r' \1 ', sentence)

def two_spaces_to_one(sentence):
    return re.sub('  ', ' ', sentence)

def is_english(word, eng_char_percent=0.5):
    len_word = len(word)
    if len(re.findall('[0-9]', word)) != 0:
        return False
    else:
        if len_word == 1:
            return True
        elif len_word == 0:
            return False
        elif word in ['<START>', '<END>']:
            return True
        else:
            return (
                True if len(re.findall('[a-zA-Z]', word)) / len(word) >
                eng_char_percent else False)

def not_english_sentence(sentence, end_char_percent=0.5):
    num_eng_char = len(re.findall('[A-Za-z]', sentence))
    if num_eng_char / len(sentence) < end_char_percent:
        return True
    return False

def is_chinese(word):
    if word in ['<START>', '<END>']:
        return True
    elif len(word) == 0:
        return False
    else:
        if len(re.findall('[a-zA-Z]', word)) != 0:
            return False
        else:
            return True

def en_txt_preproc(txt):
    s_full_processed = []
    for s in txt:
        s = re.sub('\n|\xad', '', s) # remove \n
        s = re.sub('“', '"', s)
        s = re.sub('‘', '\'', s)
        s = re.sub('\xa0|\t', ' ', s)
        s = add_space_before_punct(s) # add space before puncts
        s = s.lower() # turn into lower cases
        s = label_start_end(s) # tag start and end
        s = two_spaces_to_one(s) # if two spaces, delete one
        s_full_processed.append(s)
    return s_full_processed

def ch_txt_preproc(txt):
    s_full_processed = []
    for s in txt:
        s = re.sub('\n|\xad', '', s)
        s = re.sub('([^0-9])', r' \1 ', s)
        s = label_start_end(s)
        s = two_spaces_to_one(s)
        s = s_full_processed.append(s)
    return s_full_processed

class VocabEN(object):
    """
    Preprocess english text
    """
    def __init__(self, txt):
        self.txt = txt
        self.word_dict = None

    def create_vocab_dict(self, freq_cutoff=3, eng_char_percent=0.5):
        preprocessed_txt = en_txt_preproc(self.txt)
        ls_words = []
        for s in preprocessed_txt:
            s = s.split(' ')
            ls_words += s

        unique_words, counts = np.unique(ls_words, return_counts=True)
        sub_idx = np.where(counts >= freq_cutoff)
        unique_words_sub = unique_words[sub_idx]
        counts_sub = counts[sub_idx]
        sorted_idx = np.argsort(counts_sub)
        unique_words_sub_sorted = unique_words_sub[sorted_idx][::-1]

        final_word_ls = []
        for w in unique_words_sub_sorted:
            if is_english(w, eng_char_percent):
                final_word_ls.append(w)
        self.word_dict = dict(zip(range(2, len(final_word_ls)+2), final_word_ls))
        self.word_dict[1] = '<UNK>'
        self.rev_word_dict = {val:key for key, val in self.word_dict.items()}

    def word_to_idx(self, word):
        try:
            return self.rev_word_dict[word]
        except KeyError:
            return self.rev_word_dict['<UNK>']

    def idx_to_word(self, idx):
        return self.word_dict[idx]

class VocabCH(object):
    """
    Preprocess Chinese text
    """
    def __init__(self, txt):
        self.txt = txt

    def create_vocab_dict(self, freq_cutoff=2):
        preprocessed_txt = ch_txt_preproc(self.txt)
        ls_words = []
        for s in preprocessed_txt:
            s = s.split(' ')
            ls_words += s
        unique_words, counts = np.unique(ls_words, return_counts=True)
        sub_idx = np.where(counts >= freq_cutoff)
        unique_words_sub = unique_words[sub_idx]
        counts_sub = counts[sub_idx]
        sorted_idx = np.argsort(counts_sub)
        unique_words_sub_sorted = unique_words_sub[sorted_idx][::-1]

        final_word_ls = []
        for w in unique_words_sub_sorted:
            if is_chinese(w):
                final_word_ls.append(w)
        self.word_dict = dict(zip(range(2, len(final_word_ls)+2), final_word_ls))
        self.word_dict[1] = '<UNK>'
        self.rev_word_dict = {val:key for key, val in self.word_dict.items()}

    def word_to_idx(self, word):
        try:
            return self.rev_word_dict[word]
        except KeyError:
            return self.rev_word_dict['<UNK>']

    def idx_to_word(self, idx):
        return self.word_dict[idx]

class VocabHub(object):

    def __init__(self):
        self.en = None
        self.ch = None

    def build(self, en_txt, ch_txt, en_freq_cutoff=3, ch_freq_cutoff=2):
        self.en = VocabEN(en_txt)
        self.ch = VocabCH(ch_txt)
        self.en.create_vocab_dict(en_freq_cutoff)
        self.ch.create_vocab_dict(ch_freq_cutoff)

    def load(self, load_from_path='./vocab'):
        with open(os.path.join(load_from_path, 'vocab_hub.pkl'), 'rb') as f:
            vocab_hub = pickle.load(f)
        self.en = vocab_hub.en
        self.ch = vocab_hub.ch

    def save(self, save_to_path='./vocab'):
        create_path(save_to_path)
        with open(os.path.join(save_to_path, 'vocab_hub.pkl'), 'wb') as f:
            pickle.dump(self, f)
