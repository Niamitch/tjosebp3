import nltk
import os
nltk.download('punkt')
import numpy as np
from natsort import natsorted, ns
import sklearn
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

from keras.layers import Input, Dense, Embedding
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.utils import to_categorical

import tensorflow
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
pd.options.mode.chained_assignment = None

MAX_NB_WORDS = 80000


def load_corpus():
    features_sets = []
    maxlength = 0
    with open("./data/train.txt", encoding = "ISO-8859-1") as f:
        next(f)
        for line in f:
            tokens = nltk.word_tokenize(line, 'english')
            sentences = tokens[1:len(tokens)-2]
            if len(sentences) > maxlength:
                maxlength = len(sentences)
            label = tokens[len(tokens)-1]
            features_sets.append((sentences, getIntLabel(label)))
    return features_sets, maxlength

def getIntLabel(label):
    array = ["angry","sad","happy","others"]
    return array.index(label)

def create_rnn(maxlength):
    embedding_dim = 300
    embedding_matrix = np.random.random((MAX_NB_WORDS, embedding_dim))

    inp = Input(shape=(maxlength,))
    x = Embedding(input_dim=MAX_NB_WORDS, output_dim=embedding_dim, input_length=maxlength,
                  weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(4, activation="softmax")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def getBestModel():
    files = os.listdir("./models/rnn/")
    new_order = natsorted(files, alg=ns.IGNORECASE)
    return new_order[len(new_order)-1]

def choiseHighest(probabilities):
    for line in probabilities:
        indexOfHighest = 0;
        valueOfHighest = 0;
        for i in range(len(line)):
            if line[i] > valueOfHighest:
                indexOfHighest = i
                valueOfHighest = line[i]
        for i in range(len(line)):
            if i == indexOfHighest:
                line[i] = 1
            else:
                line[i] = 0
    return probabilities



def main():
    features_sets, maxlength = load_corpus()
    token, label = zip(*features_sets)
    label = to_categorical(label)
    x_train_set, x_test_set, y_train_set, y_test_set = sklearn.model_selection.train_test_split(token,label,test_size=0.2,random_state=42)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,lower=True)
    tokenizer.fit_on_texts(token)

    padded_train_sequences = pad_sequences(tokenizer.texts_to_sequences(x_train_set), maxlen= maxlength)
    padded_test_sequences = pad_sequences(tokenizer.texts_to_sequences(x_test_set), maxlen= maxlength)

    rnn_gru_model = create_rnn(maxlength)

    batch_size = 256
    epochs = 30

    filepath = "./models/rnn/weights-improvement-{val_acc:.4f}-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    history = rnn_gru_model.fit(x=padded_train_sequences,
                                   y=y_train_set,
                                   validation_data=(padded_test_sequences, y_test_set),
                                   batch_size=batch_size,
                                   callbacks=[checkpoint],
                                   epochs=epochs,
                                   verbose=1)

    best_rnn_simple_model = load_model('./models/rnn/'+getBestModel())

    y_pred_rnn_simple = best_rnn_simple_model.predict(padded_test_sequences, verbose=1, batch_size=2048)
    y_pred_rnn_simple = choiseHighest(y_pred_rnn_simple)
    y_pred_rnn_simple = pd.DataFrame(y_pred_rnn_simple, columns=["angry","sad","happy","others"])
    y_pred_rnn_simple.to_csv('./predictions/y_pred_rnn_simple.csv', index=False)

    y_pred_rnn_simple = pd.read_csv('./predictions/y_pred_rnn_simple.csv')
    print(accuracy_score(y_test_set, y_pred_rnn_simple))


if __name__ == "__main__":
   main()