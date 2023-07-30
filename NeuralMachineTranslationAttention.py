from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import Preprocessing as prep
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Attention
from keras.utils import pad_sequences, to_categorical

def init_glove(num_vocabs, word_index):
    """
    The following function is a modified copy of Code Example "Using pre-trained word embeddings", authored by François Chollet 
    (https://twitter.com/fchollet), URL: https://keras.io/examples/nlp/pretrained_word_embeddings/ Date created: 2020/05/05 Last modified: 2020/05/05
    Last accessed: 2023/07/29
    """
    path_to_glove_file = "/home/sherif/git/non-repository-misc/glove.6B/glove.6B.100d.txt"

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))
    num_tokens = num_vocabs + 2
    embedding_dim = 100
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Chars not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d word (%d misses)" % (hits, misses))

    return num_tokens, embedding_dim, embedding_matrix

def init_model(latent_dim, num_encoder_tokens, num_decoder_tokens, encoder_embedding_matrix, decoder_embedding_matrix, embedding_size):
    """
    Function in order to define several models with different hyper-parameters
    """
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(num_decoder_tokens, return_state=True, return_sequences=True) #TODO: Embed
    print(encoder_embedding_matrix.shape)
    #encoder_embed_layer = Embedding(num_encoder_tokens,
    #                                embedding_size, 
    #                                embeddings_initializer=tf.keras.initializers.Constant(encoder_embedding_matrix), 
    #                                trainable=False)(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder = LSTM(num_decoder_tokens, return_state=True, return_sequences=True)
    #decoder_embed_layer = Embedding(num_decoder_tokens,
    #                                embedding_size, 
    #                                embeddings_initializer=tf.keras.initializers.Constant(decoder_embedding_matrix), 
    #                                trainable=False)(decoder_inputs)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h, state_c])

    decoder_attention = Attention()
    decoder_attention_outputs = decoder_attention([encoder_outputs,decoder_outputs])

    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_attention_outputs)


    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


def train_and_test_model(model, encoder_data, decoder_data, name):

    decoder_input_data = decoder_data[:, :-1]
    decoder_target_data = decoder_data[:, 1:]
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
    
    model.fit([encoder_data, decoder_input_data],
        decoder_target_data, 
        batch_size=32,
        epochs=30,
        validation_split=0.2,
    )

    model.save(name)

def main():
    cz_sent_full = [tuple[0] for tuple in prep.data_processed]
    cz_sent = cz_sent_full[:100]
    en_sent_full = [tuple[1] for tuple in prep.data_processed]
    en_sent = en_sent_full[:100]
    test_set_size = 0.2
    #cz_train, cz_test , en_train, en_test = train_test_split(cz_sent, en_sent, test_size=test_set_size)

    cz_train, cz_test , en_train, en_test = train_test_split(en_sent, cz_sent, test_size=test_set_size)


    cz_tokenizer = Tokenizer()
    cz_tokenizer.fit_on_texts(cz_train)
    cz_seq = cz_tokenizer.texts_to_sequences(cz_train)
    max_cz_seq = max([len(seq) for seq in cz_seq])
    vocabs_cz = len(cz_tokenizer.word_index) + 1

    en_tokenizer = Tokenizer()
    en_tokenizer.fit_on_texts(en_train)
    en_seq = en_tokenizer.texts_to_sequences(en_train)
    max_en_seq = max([len(seq) for seq in en_seq])
    vocabs_en = len(en_tokenizer.word_index) + 1

    max_seq = max([max_cz_seq, max_en_seq])

    en_padded_inputs_train_1 = pad_sequences(en_seq, maxlen=max_seq ,padding="post")
    cz_padded_inputs_train_1 =  pad_sequences(cz_seq, maxlen=max_seq + 1,padding="post")

    en_padded_inputs_train_2 = pad_sequences(en_seq, maxlen=max_seq + 1,padding="post")
    cz_padded_inputs_train_2 =  pad_sequences(cz_seq, maxlen=max_seq,padding="post")
    

    en_data_1 = to_categorical(en_padded_inputs_train_1)
    cz_data_1 = to_categorical(cz_padded_inputs_train_1)

    en_data_2 = to_categorical(en_padded_inputs_train_2)
    cz_data_2 = to_categorical(cz_padded_inputs_train_2)



    #print(cz_padded_inputs_train)
    #print(en_padded_inputs_train)

    latent_dim = 64
    embedding_size = 100

    gl_e_nt1, gl_e_ed1, gl_e_emb1=init_glove(num_vocabs=vocabs_en, word_index=en_tokenizer.word_index)
    gl_t_nt1, gl_t_ed1, gl_t_emb1=init_glove(num_vocabs=vocabs_cz, word_index=cz_tokenizer.word_index)
    gl_e_nt2, gl_e_ed2, gl_e_emb2=init_glove(num_vocabs=vocabs_en, word_index=en_tokenizer.word_index)
    gl_t_nt2, gl_t_ed2, gl_t_emb2=init_glove(num_vocabs=vocabs_cz, word_index=cz_tokenizer.word_index)

    model_en_cz = init_model(latent_dim=latent_dim, 
                             embedding_size= embedding_size, 
                             num_encoder_tokens= vocabs_en,
                             num_decoder_tokens= vocabs_cz,
                             encoder_embedding_matrix=gl_e_emb1,
                             decoder_embedding_matrix=gl_t_emb1)
    
    model_cz_en = init_model(latent_dim=latent_dim, 
                             embedding_size= embedding_size, 
                             num_encoder_tokens= vocabs_cz,
                             num_decoder_tokens= vocabs_en,
                             encoder_embedding_matrix=gl_e_emb2,
                             decoder_embedding_matrix=gl_t_emb2)


    train_and_test_model(model_en_cz, en_data_1, cz_data_1, "EN_CZ")
    train_and_test_model(model_cz_en, cz_data_2, en_data_2, "CZ_EN")

    predictions_en_cz = model_en_cz.predict(en_test, cz_test)
    predictions_cz_en = model_cz_en.predict(cz_test, en_test)



if __name__ == "__main__":
    main()