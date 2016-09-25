import numpy as np

from utils import read_tac, accuracy
from sklearn import svm, linear_model, naive_bayes

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from itertools import product, izip_longest


# You should experiment with the complexity of the N-gram features
# (i.e., unigrams, or unigrams and bigrams)

# TODO: make this way more generic
param_values = {
    "n_range"  : [(1,1), (1,2), (2,2)],
    "lowercase" : [True, False],
    "nfeats" : [100, 300, 500, 1000],
    "remove_stopwords" : [True, False],
    "lemmatize" : [True, False],
    "min_df" : [1, 5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

combos = [dict(izip_longest(param_values, v)) for v in product(*param_values.values())]



# whether to distinguish upper and lower case, whether to remove stop words, etc.
# NLTK contains a list of stop words in English

# Also, remove infrequently occurring words and bigrams as features.
# You may tune the threshold at which to remove infrequent words and bigrams.

# for param in param_values.keys():
# Hold constant and modify all the others, this should be recursive?
# for value in param_values[param]:
# print value
# X,Y = read_tac(year, ):
# TODO: read_tac with different params

# training set

results = []
algos = [linear_model.LogisticRegression(), svm.LinearSVC(), naive_bayes.MultinomialNB()]

for algo in algos:
    for extra_params in combos:
        print("Running with params:")
        print(extra_params)
        print("Creating features from training dataset...")
        X,Y,count_vect = read_tac("2010", **extra_params)
        # import pdb; pdb.set_trace()
        print("Features created and data loaded for training dataset...")

        n_folds = 5
        k_fold = KFold(n=X.shape[0], n_folds=n_folds)
        scores = []
        # confusion = np.array([[0, 0], [0, 0]])
        accuracies = []

        i = 1
        for train_indices, test_indices in k_fold:

            print("Training on k-fold for k=%s\n" % i)
            # import pdb; pdb.set_trace()
            i+=1

            train_batch_x = X[train_indices]
            train_batch_y = Y[train_indices]

            test_batch_x = X[test_indices]
            test_batch_y = Y[test_indices]

            algo.fit(train_batch_x, train_batch_y)
            predictions = algo.predict(test_batch_x)

            # confusion += confusion_matrix(test_batch_y, predictions)
            # score = f1_score(test_batch_y, predictions)
            # scores.append(score)
            accuracies.append(accuracy(test_batch_y, predictions))

        print('Total docs classified:', len(X))
        # print('Score:', sum(scores)/len(scores))
        validation_acc =  (sum(accuracies)/float(len(accuracies)))
        print('K-Fold Avg. Accuracy:', validation_acc)
        # print('Confusion matrix:')
        # print(confusion)

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

    # import pdb; pdb.set_trace()
    print(results)

    import csv

    myfile = open("all_runs_%s" % type(algo).__name__, 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for result in results:
        wr.writerow(result)
