import pickle

from keras.models import Sequential, load_model
from keras.layers import Dense
import sys

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

import utils
import random
import numpy as np

# Performs classification using Logistic Regression.

FREQUENCY_DIST_FILE = '../dataset/senti140/subtrainfc-processed-freqdist.pkl'
BIFREQUENCY_DIST_FILE = '../dataset/senti140/subtrainfc-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_DATASET = '../dataset/senti140/subtrainfc-processed.csv'
TEST_PROCESSED_DATASET = '../dataset/senti140/subtestfc-processed.csv'

TRAIN_MODE = True
UG_SIZE = 15000
VOCAB_SIZE = UG_SIZE
USE_BGS = True
if USE_BGS:
    BG_SIZE = 10000
    VOCAB_SIZE = UG_SIZE + BG_SIZE
FEAT_TYPE = 'frequency'
np.random.seed(1337)
ugs = utils.top_n_words(FREQUENCY_DIST_FILE, UG_SIZE)
if USE_BGS:
    bigrams = utils.top_n_bigrams(BIFREQUENCY_DIST_FILE, BG_SIZE)


def extractFeatureVector(tweet):
    uni_Fvector = []
    bi_Fvector = []
    words = tweet.split()
    for i in range(len(words) - 1):
        word = words[i]
        nWord = words[i + 1]
        if ugs.get(word):
            uni_Fvector.append(word)
        if USE_BGS:
            if bigrams.get((word, nWord)):
                bi_Fvector.append((word, nWord))
    if len(words) >= 1:
        if ugs.get(words[-1]):
            uni_Fvector.append(words[-1])
    return uni_Fvector, bi_Fvector


def extractFeatures(tweets, batch_size=500, test_file=True, feat_type='presence'):
    num_batches = int(np.ceil(len(tweets) / float(batch_size)))
    for i in range(num_batches):
        batch = tweets[i * batch_size: (i + 1) * batch_size]
        features = np.zeros((batch_size, VOCAB_SIZE))
        labels = np.zeros(batch_size)
        for j, tweet in enumerate(batch):
            if test_file:
                tweet_words = tweet[1][0]
                tweet_bigrams = tweet[1][1]
            else:
                tweet_words = tweet[2][0]
                tweet_bigrams = tweet[2][1]
                labels[j] = tweet[1]
            if feat_type == 'presence':
                tweet_words = set(tweet_words)
                tweet_bigrams = set(tweet_bigrams)
            for word in tweet_words:
                idx = ugs.get(word)
                if idx:
                    features[j, idx] += 1
            if USE_BGS:
                for bigram in tweet_bigrams:
                    idx = bigrams.get(bigram)
                    if idx:
                        features[j, UG_SIZE + idx] += 1
        yield features, labels


def process_tweets(csv_file, test_file=True):
    tweets = []
    print('Generating feature vectors using the data')
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, sentiment, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = extractFeatureVector(tweet)
            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
            utils.write_status(i + 1, total)
    print('\n')
    return tweets


def applyTFIDF(X):
    transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
    transformer.fit(X)
    return transformer


def process_tweet(tweet, test_file=True):
    tweets = []
    # print('Generating feature vectors using the data')

    tweet_id, sentiment, tweet = tweet.split(',')
    feature_vector = extractFeatureVector(tweet)
    tweets.append((tweet_id, int(sentiment), feature_vector))
    print('\n')
    return tweets


def predictSingleTweet(clf, SampleTweet):
    tweets = process_tweet(SampleTweet, test_file=False)

    print()
    # print('\n')
    # print('Testing the data using decision')
    val_tweets = tweets
    correct, total = 0, len(val_tweets)
    i = 1
    batch_size = len(val_tweets)
    for val_set_X, val_set_y in extractFeatures(val_tweets, test_file=False, feat_type=FEAT_TYPE,
                                                batch_size=batch_size):
        # if FEAT_TYPE == 'frequency':
        #     val_set_X = tfidf.transform(val_set_X)
        prediction = clf.predict(val_set_X)
        # print(prediction)
        # pred_prob = max(model._predict_proba_lr(val_set_X)[0])
        # print(prediction)
        # print(pred_prob)
        correct += np.sum(prediction == val_set_y)
        i += 1
        # print('\nCorrect: %d/%d = %.4f %%' % (correct, total, correct * 100. / total))
        # print(prediction[0],val_set_y[0])
        return prediction[0], val_set_y[0]


if __name__ == '__main__':
    tweets = process_tweets(TRAIN_PROCESSED_DATASET, test_file=False)
    train_tweets, val_tweets = utils.split_data(tweets)

    print('Training Phase(Extracting features using logistic regression)')
    clf = LogisticRegression(multi_class='multinomial', random_state=1)
    batch_size = len(train_tweets)
    i = 1
    n_train_batches = int(np.ceil(len(train_tweets) / float(batch_size)))
    for training_set_X, training_set_y in extractFeatures(train_tweets, test_file=False, feat_type=FEAT_TYPE,
                                                          batch_size=batch_size):
        utils.write_status(i, n_train_batches)
        i += 1
        if FEAT_TYPE == 'frequency':
            tfidf = applyTFIDF(training_set_X)
            training_set_X = tfidf.transform(training_set_X)
        clf.fit(training_set_X, training_set_y)
    print('\n')
    print('Testing the data using logistic regression')
    correct, total = 0, len(val_tweets)
    i = 1
    batch_size = len(val_tweets)
    n_val_batches = int(np.ceil(len(val_tweets) / float(batch_size)))
    print('Predicting batches')
    for val_set_X, val_set_y in extractFeatures(val_tweets, test_file=False, feat_type=FEAT_TYPE,
                                                batch_size=batch_size):
        if FEAT_TYPE == 'frequency':
            val_set_X = tfidf.transform(val_set_X)
        prediction = clf.predict(val_set_X)
        correct += np.sum(prediction == val_set_y)
        utils.write_status(i, n_val_batches)
        i += 1
    print('\nCorrect: %d/%d = %.4f %%' % (correct, total, correct * 100. / total))
    clf.acc = correct / total
    f = open('savedModels/logistic.pickle', 'wb')
    pickle.dump(clf, f)
    f.close()
