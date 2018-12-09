"""
This example shows how to use an LSTM sentiment classification model trained using Keras in spaCy. spaCy splits the document into sentences, and each sentence is classified using the LSTM. The scores for the sentences are then aggregated to give the document score. This kind of hierarchical model is quite difficult in "pure" Keras or Tensorflow, but it's very effective. The Keras example on this dataset performs quite poorly, because it cuts off the documents so that they're a fixed size. This hurts review accuracy a lot, because people often summarise their rating in the final sentence

Prerequisites:
spacy download en_vectors_web_lg
pip install keras==2.0.9

Compatible with: spaCy v2.0.0+
"""
import nltk as nltk
import plac
import random
import pathlib
import cytoolz
import numpy
import sklearn
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import thinc.extra.datasets
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
from spacy.compat import pickle
import spacy

nltk.download('punkt')
nltk.download('stopwords')


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length=100):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length=100):
        self._model = model
        self.max_length = max_length

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            # for sent, label in zip(sentences, ys):
            #     sent.doc.sentiment += label - 0.5
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs


def train(train_texts, train_labels, dev_texts, dev_labels,
          lstm_shape, lstm_settings, lstm_optimizer, batch_size=100,
          nb_epoch=5, by_sentence=True):
    print("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              epochs=nb_epoch, batch_size=batch_size)
    return model


def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'],
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


def evaluate(model_dir, texts, labels, max_length=100):
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(SentimentAnalyser.load(model_dir, nlp, max_length=max_length))

    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
        i += 1
    return float(correct) / i

def remove_stopwords(sentences):
    filtered_sentenes = []
    stop_words = set(stopwords.words('english'))
    for word in sentences:
        if word not in stop_words:
            filtered_sentenes.append(word)
    return filtered_sentenes

def load_corpus():

    labels_to_one_hot_vector = {
        "others": [1,0,0,0],
        "angry": [0,1,0,0],
        "sad": [0,0,1,0],
        "happy": [0,0,0,1]
    }

    oneHotVector = OneHotEncoder(4)

    sentences = []
    with open("./data/train.txt", encoding = "ISO-8859-1") as f:
        next(f)
        for line in f:
            tokens = nltk.word_tokenize(line, 'english')
            sentence = ''.join(tokens[1:len(tokens)-2])
            label = labels_to_one_hot_vector[tokens[len(tokens) - 1]]
            sentences.append((sentence, label))
    return zip(*sentences)


def main():
    model_dir = pathlib.Path("./model/")
    nr_hidden = 64
    max_length = 100  # Shape
    dropout = 0.5
    learn_rate = 0.01  # General NN config
    nb_epoch = 1
    batch_size = 256
    nr_class = 4

    print("Read data")
    sentences, labels = load_corpus()
    sentences = remove_stopwords(sentences)
    size = len(sentences)
    train_texts, train_labels = sentences[:int(size * 0.6)], labels[:int(size * 0.6)]
    dev_texts, dev_labels = sentences[int(size * 0.6):int(size * 0.8)], labels[int(size * 0.6):int(size * 0.8)]
    test_texts, test_labels = sentences[int(size * 0.8):], labels[int(size * 0.8):]

    train_labels = numpy.asarray(train_labels, dtype='int32')
    dev_labels = numpy.asarray(dev_labels, dtype='int32')
    lstm = train(train_texts, train_labels, dev_texts, dev_labels,
                 {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': nr_class},
                 {'dropout': dropout, 'lr': learn_rate},
                 {},
                 nb_epoch=nb_epoch, batch_size=batch_size)
    print("Model has been trained!")

    weights = lstm.get_weights()
    if model_dir is not None:
        with (model_dir / 'model').open('wb') as file_:
            pickle.dump(weights[1:], file_)
        with (model_dir / 'config.json').open('w') as file_:
            file_.write(lstm.to_json())

    # nb_correct = evaluate(model_dir, test_texts, test_labels)
    # print("Number of correct: " + str(nb_correct) + " on " + str(len(test_texts)))


if __name__ == '__main__':
    plac.call(main)