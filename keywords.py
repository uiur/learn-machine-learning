# coding: utf-8
import sys
import glob
import MeCab
from gensim import corpora, matutils
import random

from sklearn.naive_bayes import MultinomialNB

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

# 1: dokujo
# 0: otherwise

labels = []
train_data_paths = []


dokujo_limit = 400
other_limit  = dokujo_limit / 4

dokujo_paths = glob.glob('./data/dokujo-tsushin/*.txt')
random.shuffle(dokujo_paths)
train_data_paths += dokujo_paths[:dokujo_limit]

for i in range(dokujo_limit):
    labels.append(1)

for type in ['it-life-hack', 'kaden-channel', 'movie-enter', 'sports-watch']:
    paths = glob.glob('./data/%s/*.txt' % type)
    random.shuffle(paths)
    train_data_paths += paths[:other_limit]
    for i in range(other_limit):
        labels.append(0)

train_data = []

for path in train_data_paths:
    input = open(path).read()
    bow = dictionary.doc2bow(tokenize(input))
    train_data.append(bow)

train = matutils.corpus2dense(train_data, num_terms=len(dictionary)).T

clf = MultinomialNB().fit(train, labels)

input = sys.stdin.read()

bow = dictionary.doc2bow(tokenize(input))
v = matutils.corpus2dense([bow], num_terms=len(dictionary)).T

print clf.predict(v)
