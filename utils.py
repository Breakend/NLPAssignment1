'''
Created on Jul 14, 2015

Modified from starter code provided for COMP 599 course

@author: jcheung
@author: peter.henderson
'''

import os
import codecs
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# TODO: add tests

stoplist = set(stopwords.words('english'))

def read_tac(year, n_range = (1,1), remove_stopwords = False, min_df=0., lemmatize = False, lowercase = True, nfeats=100):
    """Read data set and return feature matrix X and labels y.

    Args:
        year: The dataset year to grab (assumes datapath is: ../data/tac<year>)
        n_range: the range of n-grams to use as a tuple (1,2)
                 would use both 1-gram and 2-gram
        lemmatize: whether to lemmatize the words
        lowercase: whether to lowercase the words

    Returns:
        (X, Y) where X is the feature list and Y is the document
        classification label

        The return values have dimensions:
            X - (ndocs x nfeats)
            Y - (ndocs)

    """
    # modify this according to your directory structure
    sub_folder = '../data/tac%s' % year
    X, Y = [], []

    # labels
    labels_f = 'tac%s.labels' % year

    fh = open(os.path.join(sub_folder, labels_f))
    for line in fh:
        docid, label = line.split()
        Y.append(int(label))

    # tac 10
    if year == '2010':
        template = 'tac10-%04d.txt'
        s, e = 1, 921
    elif year == '2011':
        template = 'tac11-%04d.txt'
        s, e = 921, 1801

    if remove_stopwords:
        stop_words = stopwords
    else:
        stop_words = None

    count_vect = CountVectorizer(tokenizer=CustomTokenizer(lemmatize=lemmatize),
                                 stop_words=stop_words,
                                 lowercase=lowercase,
                                 ngram_range=n_range,
                                 max_features=100,
                                 min_df=min_df)


    # TODO: This is a little funky since I'm loading everything into memory
    #       do it better?

    documents = []
    # one pass to generate a vocabulary
    for i in xrange(s, e):
        fname = os.path.join(sub_folder, template % i)
        s = codecs.open(fname, 'r', encoding = 'utf-8').read()
        s = codecs.encode(s, 'ascii', 'ignore')
        documents.extend(s)

    count_vect.fit(documents)

    for doc in documents:
        X.append(extract_features(doc, count_vect))

    Y = np.array(Y)
    X = np.vstack(tuple(X))
    return X, Y

def ispunct(some_string):
    """ Function for checking if a string has punctuation

    Args:
       some_string: word to check

    Returns:
        boolean whether contains punctuation
    """
    return not any(char.isalnum() for char in some_string)

class CustomTokenizer(object):
    def __init__(self, lemmatize=False):
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

    def __call__(self, s):
        tokens = get_tokens(s)

        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

def get_tokens(s):
    """Tokenize into words in sentences.

    Args:
        s: a sentence string

    Returns:
        a list of strs
    """
    retval = []
    sents = sent_tokenize(s)

    for sent in sents:
        tokens = word_tokenize(sent)
        retval.extend(tokens)
    return retval

def extract_features(doc, count_vect):
    """Extract features from text file f into a feature vector.

    Args:
        f: file containing strings
        count_vect: (CountVectorizer) the count vectorizer to track the dictionary

    Returns:
        a list of indices?
    """
    return count_vect.transform(doc).toarray()

# evaluation code
def accuracy(gold, predict):
    """Calculates the accuracy of predicted values

    Args:
        gold: the ground truth
        predict: your predictions

    Returns:
        a float accuracy value
    """
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if int(gold[i]) == int(predict[i]):
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)
