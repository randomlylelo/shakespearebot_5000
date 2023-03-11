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

# %%
# data = pd.read_csv(SHAKESPEARE_DATA, sep=" ", header=None)
# # data.columns = ["a", "b", "c", "etc."]
# print(data.head())

def has_digit(string):
    return any(char.isdigit() for char in string)


def remove_punctuation(string):
    return re.sub(r'[^\w\s]', '', string)

with open(SHAKESPEARE_DATA, 'r') as f:
    all_data = f.readlines()
    all_lines = list(filter(lambda line: not has_digit(line), all_data))

    def clean_line(line):
        words = line.split(' ')

        words_no_punc = []

        for word in words:
            # Remove punctuation
            words_no_punc.append(remove_punctuation(word).strip())

        return words_no_punc

    X = list(map(clean_line, all_lines))

    X, X_map = parse_observations(X)

    print(X)

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
