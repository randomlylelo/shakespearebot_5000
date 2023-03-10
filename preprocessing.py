# 
# %% Download the CMU dict
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'data/Syllable_dictionary.txt'

data = pd.read_fwf(DATA_PATH, widths=10)


#%%

len(cmudict.dict())
