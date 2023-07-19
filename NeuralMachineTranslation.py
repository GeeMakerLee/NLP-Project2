from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import Preprocessing as prep
import tensorflow as tf
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
#USE SIMPLERNN <- most simple one
cz_sent = [tuple[0] for tuple in prep.data_processed]
en_sent = [tuple[1] for tuple in prep.data_processed]
test_set_size = 0.2
#cz_train, cz_test , en_train, en_test = train_test_split(cz_sent, en_sent, test_size=test_set_size)

cz_tokenizer = Tokenizer()
cz_tokenizer.fit_on_texts(cz_sent)
cz_seq = cz_tokenizer.texts_to_sequences(cz_sent)

en_tokenizer = Tokenizer()
en_tokenizer.fit_on_texts(en_sent)
en_seq = en_tokenizer.texts_to_sequences(en_sent)

cz_train, cz_test , en_train, en_test = train_test_split(cz_seq, en_seq, test_size=test_set_size)



def init_model():
    pass


def train_and_test_model():
    pass



