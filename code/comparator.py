import pickle
from collections import defaultdict
import logisticdefault
import naivebayes
import randomforest
import voting2

total = 0
predictions = defaultdict(int)

print("naive bayes")
f = open('finalTrained/naive.pickle', 'rb')
naive_clf = pickle.load(f)
f.close()

print("random forest")
f = open('finalTrained/randomforest.pickle', 'rb')
randomForest_clf = pickle.load(f)
f.close()

print("logistic")
f = open('finalTrained/logistic.pickle', 'rb')
logistic_clf = pickle.load(f)
f.close()
print("voting")
f = open('finalTrained/voting.pickle', 'rb')
voting_clf = pickle.load(f)
f.close()

filepath='../dataset/senti140/subtestfc-processed.csv'
num_lines = sum(1 for line in open(filepath))
with open(filepath) as f:
    c=0
    for tweet in f:
        c+=1
        print(str(c)+"/"+str(num_lines))
        naivePrediction, actual_label=naivebayes.predictSingleTweet(naive_clf,tweet)
        if(naivePrediction==actual_label):
            predictions['Using Naive Bayes']+=1

        randomForestPrediction, actual_label=randomforest.predictSingleTweet(randomForest_clf,tweet)
        if(randomForestPrediction==actual_label):
            predictions['Using Random Forest']+=1


        logisticRegressionPrediction,actual_label=logisticdefault.predictSingleTweet(logistic_clf ,tweet)
        if(logisticRegressionPrediction==actual_label):
            predictions['Using Logistic regression']+=1

        votingPrediction, actual_label=voting2.predictSingleTweet(voting_clf,tweet)
        if(votingPrediction==actual_label):
            predictions['Using Voting algorithm']+=1
        total+=1
for name,correct in predictions.items():
    print(name+":"+str((correct/total)*100))