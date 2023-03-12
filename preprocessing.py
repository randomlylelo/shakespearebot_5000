# %% Download the CMU dict
from hw6_code_that_we_use import parse_observations, unsupervised_HMM, sample_sentence
from nltk.corpus import cmudict
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

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
# for _ in range(14):
#   print(sample_sentence(HMM, X_map, n_words=10))

# %%
rhyme_groups = {}
for _ in tqdm(range(1000)):
    line = sample_sentence(HMM, X_map, n_words=10)
    word_list = line.split(" ")
    last_word = word_list[len(word_list) - 1]
    if (len(last_word) < 2):
        continue

    last_2_letters = last_word[len(last_word)-2:]
    if last_2_letters not in rhyme_groups:
        rhyme_groups[last_2_letters] = []
    rhyme_groups[last_2_letters].append(line)  # getting the last 2 letters and using it as a dictionary key

print(rhyme_groups)



#%%
# get rhyming to work
to_print = []
last_word_set = set()

for key in rhyme_groups:
    lst = rhyme_groups[key]
    if len(lst) > 2:
        counter = 0
        # deal with the case where there is no two
        to_add_to_print = []
        for sentence in lst:
            last_word = sentence.split()[-1]
            if last_word not in last_word_set:
                counter += 1
                to_add_to_print.append(sentence)
                last_word_set.add(last_word)

            if counter == 2:
                for str in to_add_to_print:
                    to_print.append(str)
                break

    if len(to_print) == 14:
        break

print(to_print[0]) # a
print(to_print[2]) # b
print(to_print[1]) # a
print(to_print[3]) # b
print()

print(to_print[4]) # c
print(to_print[6]) # d
print(to_print[5]) # c
print(to_print[7]) # d
print()

print(to_print[8]) # e
print(to_print[10]) # f
print(to_print[9]) # e
print(to_print[11]) # f
print()

print(to_print[12]) # g
print(to_print[13]) # g

#%%

# Try to get CMU rhyme dictionary
pronouce_dct = cmudict.dict()
cmu_rhyme_groups = {}
for _ in tqdm(range(1000)):
    line = sample_sentence(HMM, X_map, n_words=10)
    word_list = line.split(" ")
    last_word = word_list[len(word_list) - 1]
    if (len(last_word) < 2):
        continue

    if last_word in pronouce_dct:
        # get last couple pronounation, since we can't use list, use str rep
        pronouce = repr(pronouce_dct[last_word][0][-2:])
        if pronouce not in cmu_rhyme_groups:
            cmu_rhyme_groups[pronouce] = []
        cmu_rhyme_groups[pronouce].append(line)

print(cmu_rhyme_groups)

# to_print = []
# for key in cmu_rhyme_groups:
#     lst = cmu_rhyme_groups[key]
#     if len(lst) > 2:
#         counter = 0
#         for sentence in lst:
#             counter += 1
#             to_print.append(sentence)

#         if counter == 2:
#             continue

# %%