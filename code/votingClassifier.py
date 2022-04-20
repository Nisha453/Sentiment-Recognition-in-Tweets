# import pickle
# import numpy as np
# from sklearn.feature_extraction.text import TfidfTransformer
# import naivebayes
# import utils
# from naivebayes import process_tweets, extract_features
#
#
# f = open('savedModels/svm.pickle', 'rb')
# svmClf = pickle.load(f)
# f.close()
#
# def naivepredictSingleTweet(tweet):
#     f = open('savedModels/naive.pickle', 'rb')
#     naiveClf = pickle.load(f)
#     f.close()
#     np.random.seed(1337)
#     unigrams = utils.top_n_words(naivebayes.FREQ_DIST_FILE, naivebayes.UNIGRAM_SIZE)
#     if naivebayes.USE_BIGRAMS:
#         bigrams = utils.top_n_bigrams(naivebayes.BI_FREQ_DIST_FILE, naivebayes.BIGRAM_SIZE)
#
#     def get_feature_vector(tweet):
#         uni_feature_vector = []
#         bi_feature_vector = []
#         words = tweet.split()
#         for i in range(len(words) - 1):
#             word = words[i]
#             next_word = words[i + 1]
#             if unigrams.get(word):
#                 uni_feature_vector.append(word)
#             if naivebayes.USE_BIGRAMS:
#                 if bigrams.get((word, next_word)):
#                     bi_feature_vector.append((word, next_word))
#         if len(words) >= 1:
#             if unigrams.get(words[-1]):
#                 uni_feature_vector.append(words[-1])
#         return uni_feature_vector, bi_feature_vector
#
#     def extract_features(tweet, batch_size=1, test_file=False, feat_type='presence'):
#         features = naivebayes.lil_matrix((batch_size, naivebayes.VOCAB_SIZE))
#         labels = np.zeros(batch_size)
#         if test_file:
#             tweet_words = tweet[1][0]
#             tweet_bigrams = tweet[1][1]
#         else:
#             tweet_words = tweet[2][0]
#             tweet_bigrams = tweet[2][1]
#             labels[0] = tweet[1]
#         if feat_type == 'presence':
#             tweet_words = set(tweet_words)
#             tweet_bigrams = set(tweet_bigrams)
#         for word in tweet_words:
#             idx = unigrams.get(word)
#             if idx:
#                 features[0, idx] += 1
#         if naivebayes.USE_BIGRAMS:
#             for bigram in tweet_bigrams:
#                 idx = bigrams.get(bigram)
#                 if idx:
#                     features[0, naivebayes.UNIGRAM_SIZE + idx] += 1
#         return features, labels
#
#     fv=get_feature_vector(tweet)
#     naiveClf.predict(extract_features(fv))
#
# naivepredictSingleTweet("very good")
# # # f = open('savedModels/decision_tree.pickle', 'rb')
# # # decisionTreeClf = pickle.load(f)
# # # f.close()
# # TEST_PROCESSED_FILE = '../dataset/senti140/tsfc-processed-processed.csv'
# # FEAT_TYPE = 'frequency'
# #
# # def apply_tf_idf(X):
# #     transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
# #     transformer.fit(X)
# #     return transformer
# #
# # test_tweets = process_tweets(TEST_PROCESSED_FILE, test_file=True)
# # predictions = np.array([])
# #
# # i = 1
# # for test_set_X, _ in extract_features(test_tweets, test_file=True, feat_type=FEAT_TYPE):
# #     tfidf = apply_tf_idf(test_set_X)
# #     if FEAT_TYPE == 'frequency':
# #         test_set_X = tfidf.transform(test_set_X)
# #     prediction = naiveClf.predict(test_set_X)
# #     predictions = np.concatenate((predictions, prediction))
# #     i += 1
# # predictions = [(str(j), int(predictions[j]))
# #                for j in range(len(test_tweets))]
# # utils.save_results_to_csv(predictions, 'naivebayes.csv')
# # print('\nSaved to naivebayes.csv')
import datetime
import pickle
import time
import matplotlib.pyplot as plt
import naivebayes
import decisiontree
import csv
total=0
correct=0

timestamp=[]
accuracy=[]
weights=[]
plt.ion()

# plt.title('Accuracy over time')
# plt.xlabel('timestamp in milliseconds')
# plt.ylabel('accuracy')
# plt.show()
with open('../dataset/senti140/tsfc-processed-processed.csv') as f:
    for tweet in f:
        # if total%100==0:
        # plt.plot(timestamp,accuracy)
        # plt.pause(0.05)
        predections=[]
        probability=[]

        print("naive")
        f = open('savedModels/naive.pickle', 'rb')
        naive_clf = pickle.load(f)
        f.close()
        naivePrediction,naiveProbability,actual_label,naiveAcc=naivebayes.predictSingleTweet(naive_clf,tweet)
        # print(naivePrediction,naiveProbability,actual_label,naiveAcc)
        predections.append(naivePrediction)
        probability.append(naiveProbability)


        # print("SVM")
        # f = open('savedModels/svm.pickle', 'rb')
        # svm_clf = pickle.load(f)
        # f.close()
        # svmPrediction,svmProbability,actual_label,svmAcc=svmV2.predictSingleTweet(svm_clf,tweet)
        # # print(svmPrediction,svmProbability,actual_label)
        # predections.append(svmPrediction)
        # probability.append(svmProbability)


        print("decision")
        f = open('savedModels/decision_tree.pickle', 'rb')
        decision_clf = pickle.load(f)
        f.close()
        decisionPrediction,decisionProbability,actual_label,decisionACC=decisiontree.predictSingleTweet(decision_clf,tweet)
        # print(decisionPrediction,decisionProbability,actual_label)
        predections.append(decisionPrediction)
        probability.append(decisionProbability)
        # print("///////////////////")


        totalAcc=naiveAcc+decisionACC#+svmAcc#
        naiveweight=naiveAcc/totalAcc
        # svmWeight=svmAcc/totalAcc
        decisionWeight=decisionACC/totalAcc
        weights.append(naiveweight)
        weights.append(decisionWeight)

        pos=0
        neg=0
        posscore=0
        negscore=0
        for pred in predections:
            if pred:
                pos+=1
            else:
                neg+=1
        posprob=pos/(pos+neg)
        negprob=neg/(pos+neg)

        poscore=0
        negscore=0
        for pred,weight in zip(predections,weights):
            if pred:
                poscore+=(weight*posprob)
            else:
                negscore+=(weight*negprob)
        # print(negscore,poscore)
        finpred=0 if negscore>poscore else 1
        # print(finpred)
        if finpred==actual_label:
            correct+=1
        total+=1
        print("total:"+str(total))
        print("Accuracy:" + str(correct/total))
        accuracy.append(correct/total)
        timestamp.append(round(time.time() * 1000))
print(correct,total,correct/total)

