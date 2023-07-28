from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import Preprocessing as prep
import tensorflow as tf
import spacy
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

"""
    REFERENCES:
    The following section is a modified copy of Code Example "Character-level recurrent sequence-to-sequence model", authored by François Chollet 
    (https://twitter.com/fchollet), URL: https://keras.io/examples/nlp/lstm_seq2seq/ Date created: 2017/09/29, Last modified: 2020/04/26
    Last accessed: 2023/07/28
"""

#nlp = spacy.load("en_core_web_sm")

#def preprocess_again(texts):
    

def initialize_np_arrays(input_texts, target_texts):
    """
    The following function is a modified copy of Code Example "Character-level recurrent sequence-to-sequence model", authored by François Chollet 
    (https://twitter.com/fchollet), URL: https://keras.io/examples/nlp/lstm_seq2seq/ Date created: 2017/09/29, Last modified: 2020/04/26
    Last accessed: 2023/07/28
    """
    input_characters = sorted(list("".join(input_texts)))
    target_characters = sorted(list("".join(target_texts)))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

        #TODO output!
    return encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens

def init_model(latent_dim, embedding_size, num_encoder_tokens, num_decoder_tokens):
    """
    Function in order to define several models with different hyper-parameters
    """
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    #encoder_inputs = Input(shape=(len(enco), 1))
    encoder_embedding = Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_embedding = Embedding(num_decoder_tokens, embedding_size)(decoder_inputs)
    #decoder_inputs = Input(shape=(len(deco), 1))
    decoder = LSTM(latent_dim, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h, state_c])

    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)


    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


def train_and_test_model(model, encoder_input_data, decoder_input_data, decoder_target_data, name):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
    
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
        batch_size=32,
        epochs=20,
        validation_split=0.2,
    )

    model.save(name)

def main():
    latent_dim = 256
    embedding_size = 100


    cz_sent_full = [tuple[0] for tuple in prep.data_processed]
    cz_sent = cz_sent_full[:100]
    en_sent_full = [tuple[1] for tuple in prep.data_processed]
    en_sent = en_sent_full[:100]
    test_set_size = 0.2

    en_train, en_test, cz_train, cz_test  = train_test_split(en_sent, cz_sent, test_size=test_set_size)

    e1, d1, t1, num_e1, num_d1 = initialize_np_arrays(input_texts= en_train, #EN-CZ
                         target_texts= cz_train)
    
    e2, d2, t2, num_e2, num_d2 = initialize_np_arrays(input_texts= cz_train, #CZ-EN
                         target_texts= en_train)



    model_en_cz = init_model(latent_dim=latent_dim, 
                             embedding_size= embedding_size, 
                             num_encoder_tokens= num_e1,
                             num_decoder_tokens= num_d1)
    
    model_cz_en = init_model(latent_dim=latent_dim, 
                             embedding_size= embedding_size, 
                             num_encoder_tokens= num_e2,
                             num_decoder_tokens= num_d2)
    
    train_and_test_model(model=model_en_cz, 
                         encoder_input_data=e1, 
                         decoder_input_data=d1, 
                         decoder_target_data=t1, 
                         name="EN_CZ")
    
    train_and_test_model(model=model_cz_en, 
                     encoder_input_data=e2, 
                     decoder_input_data=d2, 
                     decoder_target_data=t2, 
                     name="EN_CZ")
    

    #predictions_en_cz = model_en_cz.predict(en_test)
    #predictions_cz_en = model_cz_en.predict(cz_test)



if __name__ == "__main__":
    main()


"""
    REFERENCES:
    This code is a modified copy of Code Example "Character-level recurrent sequence-to-sequence model", authored by François Chollet 
    (https://twitter.com/fchollet), URL: https://keras.io/examples/nlp/lstm_seq2seq/ Date created: 2017/09/29, Last modified: 2020/04/26
    Last accessed: 2023/07/28
"""