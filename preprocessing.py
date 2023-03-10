# %% Download the CMU dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

# constants
SYLLABLE_DICTIONARY_PATH = 'data/Syllable_dictionary.txt'

#%%
# testing out the CMU dict
# cmudict.dict()

syllable_dict = {}
with open(SYLLABLE_DICTIONARY_PATH, 'r') as f:
  for line in f:
    line = line.strip()

    if line:
      sentence = line.split(' ')
      word = sentence[0]
      syllable_dict[word] = sentence[1:]

# %%
SHAKESPEARE_DATA = 'data/shakespeare.txt'

# data = pd.read_csv(SHAKESPEARE_DATA, sep=" ", header=None)
# # data.columns = ["a", "b", "c", "etc."]
# print(data.head())


def has_digit(string):
    return any(char.isdigit() for char in string)


with open(SHAKESPEARE_DATA, 'r') as f:
    all_data = f.readlines()

    all_lines = list(filter(lambda line: not has_digit(line), all_data))

    print(all_lines)
    # print(all_data)
    # print(all_data[:100])
    # print(type(all_data))
