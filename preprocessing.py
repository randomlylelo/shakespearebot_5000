# %% Download the CMU dict
from hw6_code_that_we_use import parse_observations, unsupervised_HMM, sample_sentence
from nltk.corpus import cmudict
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import nltk
nltk.download('cmudict')

# constants
SYLLABLE_DICTIONARY_PATH = 'data/Syllable_dictionary.txt'
SHAKESPEARE_DATA = 'data/shakespeare.txt'
SPENSER_DATA = 'data/spenser.txt'

# %%
syllable_dict = {}
with open(SYLLABLE_DICTIONARY_PATH, 'r') as f:
    for line in f:
        line = line.strip()

        if line:
            sentence = line.split(' ')
            word = sentence[0]
            syllable_dict[word] = sentence[1:]

# TODO: Use CMUDict for rhymes
# testing out the CMU dict
# cmudict.dict()

# %% data importing
def has_roman_digit(string):
    # def is_roman(char):
    #     try:
    #         roman.fromRoman(char)
    #         # don't care lolz, if it doesn't error then it is a roman numeral
    #         return True
    #     except:
    #         return False
    # print(string)
    # print(len(string))
    # return any(is_roman(char) for char in string.split())
    # ^DOESN'T WORK SINCE TAKES CHAR BUT ALSO 'I' is technically roman
    # numeral so it breaks.

    # lol major hack, since I prepared the data, all the roman numerals
    # in blocks solo, so just take the length.
    return len(string) < 10

with open(SPENSER_DATA, 'r') as f:
    all_data_sp = f.readlines()

    # deal with the two edge cases in the data
    # why tf does there exist ":poem"
    all_data_sp = [listo for listo in all_data_sp if ':poem' not in listo]
    # "LVIII.", just give up the whole stanza, no clue wtf it is
    all_data_sp = [listo for listo in all_data_sp if 'LVIII.' not in listo]

    all_lines_sp = list(filter(lambda line: not has_roman_digit(line), all_data_sp))

# should be less bc of the white spaces
# print(len(all_lines_sp) - len(all_data_sp))

def has_digit(string):
    return any(char.isdigit() for char in string)

with open(SHAKESPEARE_DATA, 'r') as f:
    all_data_shake = f.readlines()
    all_lines_shake = list(filter(lambda line: not has_digit(line), all_data_shake))

# combine both datasets.
all_lines = all_lines_shake + all_lines_sp

# %% Data processing

def remove_punctuation(string):
    return re.sub(r'[^\w\s]', '', string)

def clean_line(line):
    words = line.split(' ')

    words_no_punc = []

    for word in words:
        # Remove punctuation
        words_no_punc.append(remove_punctuation(word).strip())

    return words_no_punc

X = list(map(clean_line, all_lines))

X, X_map = parse_observations(X)


# %%

"""
to use our HMM:
```
HMM = HiddenMarkovModel(A, O)
HMM.unsupervised_learning(X, N_iters)
```

what we'll need:
-randomly initialized A and O matrices
-X: A dataset consisting of input sequences in the form
    of variable-length lists, consisting of integers
    ranging from 0 to D - 1. In other words, a list of
    lists.


"""
num_hidden_states = 10
# more iters take a while
N_iters = 10
HMM = unsupervised_HMM(X, num_hidden_states, N_iters, seed=None)

# %%
# create sonnet
for _ in range(14):
  print(sample_sentence(HMM, X_map, n_words=10))

# %%
