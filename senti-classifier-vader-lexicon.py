import nltk
import random
import sklearn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('vader_lexicon')

def load_corpus():
    features_sets = []
    with open("./data/train.txt", encoding = "ISO-8859-1") as f:
        next(f)
        for line in f:
            tokens = nltk.word_tokenize(line, 'english')
            sentences = tokens[1:len(tokens)-2]
            sentences = remove_stopwords(sentences)
            label = tokens[len(tokens) - 1]
            features_sets.append((get_positive_negative_feature(''.join(sentences)), label))
    return features_sets

def remove_stopwords(sentences):
    filtered_sentenes = []
    stop_words = set(stopwords.words('english'))
    for word in sentences:
        if word not in stop_words:
            filtered_sentenes.append(word)
    return filtered_sentenes

def get_positive_negative_feature(sentences):
    features_count_by_sentiment = {}
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentences)
    features_count_by_sentiment[0] = score['neg']
    features_count_by_sentiment[1] = score['neu']
    features_count_by_sentiment[2] = score['pos']
    return features_count_by_sentiment

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
    print("Accuracy Score " + str(sklearn.metrics.accuracy_score(real_test_classes, predicted_test_classes)))

if __name__ == "__main__":
   main()