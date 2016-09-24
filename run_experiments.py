import numpy as np

from utils import read_tac, accuracy
from sklearn import svm, linear_model, naive_bayes

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score


# You should experiment with the complexity of the N-gram features
# (i.e., unigrams, or unigrams and bigrams)

# TODO: make this way more generic
param_values = {
    "n_range"  : [(1,1), (1,2), (2,2)],
    "lowercase" : [True, False],
    "stopwords" : [True, False],
    "lemmatize" : [True, False],
    "min_df" : [0.1, 0.2, 0.3, 0.4, 0.5]
    }

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
X,Y = read_tac("2010")

n_folds = 5
k_fold = KFold(n=len(X), n_folds=n_folds)
scores = []
confusion = np.array([[0, 0], [0, 0]])
accuracies = []
algo = linear_model.LogisticRegression()

for train_indices, test_indices in k_fold:
    train_batch_x = X[train_indices]
    train_batch_y = Y[train_indices]

    test_batch_x = X[test_indices]
    test_batch_y = Y[test_indices]

    algo.fit(train_batch_x, train_batch_y)
    predictions = algo.predict(test_batch_x)

    confusion += confusion_matrix(test_batch_y, predictions)
    score = f1_score(test_batch_y, predictions)
    scores.append(score)
    accuracies.append(accuracy(test_batch_y, predictions))

print('Total docs classified:', len(X))
print('Score:', sum(scores)/len(scores))
print('K-Fold Avg. Accuracy:', sum(accuracies)/len(accuracies))
print('Confusion matrix:')
print(confusion)

# test set
X,Y = read_tac("2011")
predictions = algo.predict(X)
print("Test Accuracy:", accuracy(Y, predictions))
