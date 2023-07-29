import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import Preprocessing as prep
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.initializers import Constant
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

    The code on the 2017/09/29 blog post authored by François Chollet
    "A ten-minute introduction to sequence-to-sequence learning in Keras"
    was also used as a reference for init_model and train_and_test_model
    URL: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    Last accessed: 2023/07/28
"""

    

def initialize_np_arrays(input_texts, target_texts):
    """
    The following function is a modified copy of Code Example "Character-level recurrent sequence-to-sequence model", authored by François Chollet 
    (https://twitter.com/fchollet), URL: https://keras.io/examples/nlp/lstm_seq2seq/ Date created: 2017/09/29, Last modified: 2020/04/26
    Last accessed: 2023/07/28
    """
    input_characters, target_characters = [], []
    for i in input_texts:
        for j in i:
            input_characters.append(j)
    input_characters = sorted(input_characters)
    for i in target_texts:
        for j in i:
            target_characters.append(j)
    target_characters = sorted(target_characters)
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)

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
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0
    return encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index, input_characters, target_characters

def init_glove(voc, char_index):
    """
    The following function is a modified copy of Code Example "Using pre-trained word embeddings", authored by François Chollet 
    (https://twitter.com/fchollet), URL: https://keras.io/examples/nlp/pretrained_word_embeddings/ Date created: 2020/05/05 Last modified: 2020/05/05
    Last accessed: 2023/07/29
    """
    path_to_glove_file = "/home/sherif/git/non-repository-misc/glove.6B/glove.6B.100d.txt"

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            char, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[char] = coefs

    print("Found %s char vectors." % len(embeddings_index))
    num_tokens = len(voc) + 2
    embedding_dim = 100
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for char, i in char_index.items():
        embedding_vector = embeddings_index.get(char)
        if embedding_vector is not None:
            # Chars not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d chars (%d misses)" % (hits, misses))

    return num_tokens, embedding_dim, embedding_matrix



def init_model(latent_dim, embedding_size, num_encoder_tokens, num_decoder_tokens, 
               num_tokens, encoder_embedding_matrix, decoder_embedding_matrix):
    """
    Function in order to define several models with different hyper-parameters
    """

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True) #TODO: Embed
    encoder_embed_layer = Embedding(num_encoder_tokens + 2,
                                    embedding_size, 
                                    embeddings_initializer=tf.keras.initializers.Constant(encoder_embedding_matrix), 
                                    trainable=False)(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder(encoder_embed_layer)

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder = LSTM(latent_dim, return_state=True, return_sequences=True)
    decoder_embed_layer = Embedding(num_decoder_tokens + 2,
                                    embedding_size, 
                                    embeddings_initializer=tf.keras.initializers.Constant(decoder_embedding_matrix), 
                                    trainable=False)(decoder_inputs)
    decoder_outputs, _, _ = decoder(decoder_embed_layer, initial_state=[state_h, state_c])

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


    #The following section is a modified copy of Code Example "Character-level recurrent sequence-to-sequence model", authored by François Chollet 
    #(https://twitter.com/fchollet), URL: https://keras.io/examples/nlp/lstm_seq2seq/ Date created: 2017/09/29, Last modified: 2020/04/26
    #Last accessed: 2023/07/28

    e1, d1, t1, num_e1, num_d1, idex1, tdex1, vi1, vt1 = initialize_np_arrays(input_texts= en_train, #EN-CZ
                         target_texts= cz_train)
    
    e2, d2, t2, num_e2, num_d2, idex2, tdex2, vi2, vt2 = initialize_np_arrays(input_texts= cz_train, #CZ-EN
                         target_texts= en_train)

    gl_e_nt1, gl_e_ed1, gl_e_emb1=init_glove(voc=vi1, char_index=idex1)
    gl_t_nt1, gl_t_ed1, gl_t_emb1=init_glove(voc=vt1, char_index=tdex1)
    gl_e_nt2, gl_e_ed2, gl_e_emb2=init_glove(voc=vi2, char_index=idex2)
    gl_t_nt2, gl_t_ed2, gl_t_emb2=init_glove(voc=vt2, char_index=tdex2)

    model_en_cz = init_model(latent_dim=latent_dim, 
                             embedding_size= embedding_size, 
                             num_encoder_tokens= num_e1,
                             num_decoder_tokens= num_d1,
                             num_tokens=gl_e_nt1,
                             encoder_embedding_matrix=gl_e_emb1,
                             decoder_embedding_matrix=gl_t_emb1)
    
    model_cz_en = init_model(latent_dim=latent_dim, 
                             embedding_size= embedding_size, 
                             num_encoder_tokens= num_e2,
                             num_decoder_tokens= num_d2,
                             num_tokens=gl_e_nt2,
                             encoder_embedding_matrix=gl_e_emb2,
                             decoder_embedding_matrix=gl_t_emb2)
    
    train_and_test_model(model=model_en_cz, 
                         encoder_input_data=e1, 
                         decoder_input_data=d1, 
                         decoder_target_data=t1, 
                         name="EN_CZ")
    
    train_and_test_model(model=model_cz_en, 
                     encoder_input_data=e2, 
                     decoder_input_data=d2, 
                     decoder_target_data=t2, 
                     name="CZ_EN")
    

    predictions_en_cz = model_en_cz.predict(en_test)
    predictions_cz_en = model_cz_en.predict(cz_test)



if __name__ == "__main__":
    main()