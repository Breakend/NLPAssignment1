'''
Code to run all expirements with various feature parameters and algorithms
on a document classification problem with the TAC 2010-2011 datasets

@author: peter.henderson
'''
from utils import read_tac, accuracy
from sklearn import svm, linear_model, naive_bayes

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from itertools import product, izip_longest
import csv

log = False

# You should experiment with the complexity of the N-gram features
# (i.e., unigrams, or unigrams and bigrams)
#
param_values = {
    "n_range"  : [(1,1), (1,2), (2,2)],
    "lowercase" : [True, False],
    "nfeats" : [100, 300, 500, 1000],
    "remove_stopwords" : [True, False],
    "lemmatize" : [True, False],
    "min_df" : [1, 5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

combos = [dict(izip_longest(param_values, v)) for v in product(*param_values.values())]

# training set

algos = [linear_model.LogisticRegression(), svm.LinearSVC(), naive_bayes.MultinomialNB()]
run_combos = False
for algo in algos:
    results = []
    for extra_params in combos:
        print("Running with params:")
        print(extra_params)
        print("Creating features from training dataset...")
        X,Y,count_vect = read_tac("2010", **extra_params)
        print("Features created and data loaded for training dataset...")

        n_folds = 5
        k_fold = KFold(n=X.shape[0], n_folds=n_folds)
        scores = []
        accuracies = []

        i = 1
        for train_indices, test_indices in k_fold:

            print("Training on k-fold for k=%s\n" % i)
            i+=1

            train_batch_x = X[train_indices]
            train_batch_y = Y[train_indices]

            test_batch_x = X[test_indices]
            test_batch_y = Y[test_indices]

            algo.fit(train_batch_x, train_batch_y)
            predictions = algo.predict(test_batch_x)

            accuracies.append(accuracy(test_batch_y, predictions))

        print('Total docs classified:', len(X))
        validation_acc =  (sum(accuracies)/float(len(accuracies)))
        print('K-Fold Avg. Accuracy:', validation_acc)

        # test set
        print("Loading test set...")
        X,Y,count_vect = read_tac("2011", test_data=True, count_vect=count_vect, **extra_params)
        predictions = algo.predict(X)
        test_acc=accuracy(Y, predictions)
        print("Test Accuracy:", test_acc)
        run_results = extra_params.values()
        run_results.append(validation_acc)
        run_results.append(test_acc)
        results.append(run_results)
        print(confusion_matrix(Y, predictions))

    print(results)

    if log:
        myfile = open("all_runs_%s" % type(algo).__name__, 'wb')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for result in results:
            wr.writerow(result)
