import nltk
import random
import sklearn
#WNAffect come from https://github.com/clemtoy/WNAffect
from WNAffect.wnaffect import WNAffect


def load_corpus():
    features_sets = []
    with open("./data/train.txt", encoding = "ISO-8859-1") as f:
        line = f.read()
        tokens = nltk.word_tokenize(line, 'english')
        sentences = tokens[1:len(tokens)-2]
        label = tokens[len(tokens-1)]
        features_sets.append((get_features_count(sentences), label))
    return features_sets

def get_features_count(tokens):
    wna = WNAffect('wordnet-1.6/', 'wn-domains-3.2/')
    #todo use affect wordnet to get sentiment for each words
    feature_counts = {}
    # for token in tokens:
    #     if token not in feature_counts:
    #         feature_counts[token] = 1
    #     else:
    #         feature_counts[token] += 1
    return feature_counts

def get_train_test_sets(features_sets):
    random.shuffle(features_sets)
    size = len(features_sets)
    train_set, test_set = features_sets[:int(size * 0.8)], features_sets[int(size * 0.8):]
    return train_set, test_set

def main():
    features_sets = load_corpus()
    train_set, test_set = get_train_test_sets(features_sets)
    classifier = nltk.NaiveBayesClassifier
    classifier = classifier.train(train_set)
    real_test_classes = []
    predicted_test_classes = []
    for test in test_set:
        real_test_classes.append(test[1])
        predicted_test_classes.append(classifier.classify(test[0]))
    print("F1 Score " + str(sklearn.metrics.f1_score(real_test_classes, predicted_test_classes)))

if __name__ == "__main__":
   main()