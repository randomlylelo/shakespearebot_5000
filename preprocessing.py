# %% Download the CMU dict
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# constants
SYLLABLE_DICTIONARY_PATH = 'data/Syllable_dictionary.txt'

#%%
# testing out the CMU dict
# cmudict.dict()

# %%

syllable_dict = {}
with open(SYLLABLE_DICTIONARY_PATH, 'r') as f:
  for line in f:
    line = line.strip()

    if line:
      sentence = line.split(' ')
      word = sentence[0]
      syllable_dict[word] = sentence[1:]

# %%

print(syllable_dict)