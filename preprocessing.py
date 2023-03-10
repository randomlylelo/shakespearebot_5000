#
# %% Download the CMU dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk.corpus import cmudict
import nltk
nltk.download('cmudict')

# %%

# DATA_PATH = 'data/Syllable_dictionary.txt'

# data = pd.read_fwf(DATA_PATH, widths=10)


# #%%

# len(cmudict.dict())


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
