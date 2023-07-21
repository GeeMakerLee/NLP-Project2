from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import Preprocessing as prep
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, Dense, SimpleRNN
from keras.utils import pad_sequences


def init_model(enco, deco, latent_dim, num_decoder_tokens):
    """
    Function in order to define several models with different hyper-parameters
    """
    encoder_inputs = Input(shape=(len(enco), 1))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(len(deco), 1))
    decoder = LSTM(latent_dim, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h, state_c])

    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)


    pass


def train_and_test_model():
    pass

def main():
    cz_sent = [tuple[0] for tuple in prep.data_processed]
    en_sent = [tuple[1] for tuple in prep.data_processed]
    test_set_size = 0.2
    #cz_train, cz_test , en_train, en_test = train_test_split(cz_sent, en_sent, test_size=test_set_size)

    cz_tokenizer = Tokenizer()
    cz_tokenizer.fit_on_texts(cz_sent)
    cz_seq = cz_tokenizer.texts_to_sequences(cz_sent)
    vocabs_cz = len(cz_tokenizer.word_index) + 1

    en_tokenizer = Tokenizer()
    en_tokenizer.fit_on_texts(en_sent)
    en_seq = en_tokenizer.texts_to_sequences(en_sent)
    vocabs_eb = len(en_tokenizer.word_index) + 1

    en_padded_inputs = pad_sequences(en_seq, padding="post")
    cz_padded_inputs =  pad_sequences(cz_seq, padding="post")

    en_input_data = np.array([en_padded_inputs])
    cz_input_data = np.array([cz_padded_inputs])

    print(en_padded_inputs)
    print(cz_padded_inputs)

    cz_train, cz_test , en_train, en_test = train_test_split(cz_seq, en_seq, test_size=test_set_size)

if __name__ == "__main__":
    main()

