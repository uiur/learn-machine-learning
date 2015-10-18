# coding: utf-8
import sys
import re
import glob
import MeCab
from gensim import corpora, matutils
import random
import numpy as np

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

mecab = MeCab.Tagger('mecabrc')

stopwords = open('./stopwords.txt').read().split("\n")

def tokenize(text):
    node = mecab.parseToNode(text)
    words = []
    while node:
        feature = node.feature.split(',')
        if feature[0] == '名詞' and feature[1] != '数':
            words.append(node.surface)
        node = node.next

    words = [word for word in words if not (word in stopwords)]

    return words

# documents = [
#     open(path).read() for path in glob.glob('./data/**/*.txt')
# ]
#
# dictionary = corpora.Dictionary([tokenize(document) for document in documents])
# dictionary.save('./test.dict')

dictionary = corpora.Dictionary.load('./test.dict')

data_paths = glob.glob('./data/**/*.txt')

def is_dokujo(path):
    if re.search("dokujo-tsushin", path):
        return 1
    else:
        return 0

labels = np.array(map(is_dokujo, data_paths))

bows = []
for path in data_paths:
    input = open(path).read()
    bow = dictionary.doc2bow(tokenize(input))
    bows.append(bow)

data = matutils.corpus2dense(bows, num_terms=len(dictionary)).T

scores = []

kf = cross_validation.KFold(len(data), n_folds=4, shuffle=True)
for train_indexes, test_indexes in kf:
    train = data[train_indexes]
    train_answers = labels[train_indexes]

    test = data[test_indexes]
    test_answers = labels[test_indexes]

    # clf = MultinomialNB().fit(train, train_answers)
    clf = svm.SVC().fit(train, train_answers)

    score = clf.score(test, test_answers)
    print score
    scores.append(score)

print "Mean: %.5f" % np.mean(scores)
