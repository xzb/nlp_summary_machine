# -*- coding: utf-8 -*-
__author__ = 'xiezebin'

import networkx as nx
import numpy as np

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer


def stemSen(sentence):
    stemmer = SnowballStemmer("english")
    words = sentence.split(" ")
    rvSen = ""
    for word in words:
        rvSen += stemmer.stem(word) + " "
    return rvSen

# class customCountVectorizer(CountVectorizer):
#     def build_preprocessor(self):
#         return stemSen
# wordCounter = customCountVectorizer(stop_words='english')                 # approach 2.b: custom class

class TextRank(object):
    def rank(self, doc = "doc.txt", out = "doc_textrank.txt", top = 10, stop_word = False, stem = False):
        with open(doc, "r") as lofile:
            document = lofile.read()

        # == refine document for process ==
        document = document.replace('\n', ' ')\
            .replace('."', '. "').replace('?"', '? "').replace('!"', '! "')\
            .replace('.”', '. ”').replace('?”', '? ”').replace('!”', '! ”')\
            .decode('utf-8')
        # document = ' '.join(document.strip().split('\n'))

        # == sentences tokenize ==
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
        sentence_tokenizer = PunktSentenceTokenizer(punkt_param)
        sentences = sentence_tokenizer.tokenize(document)


        # FEATURE: stem words, remove stop words


        # == count words for each sentence ==
        wordCounter = CountVectorizer()                                            # approach 0: non
        if stop_word:
            wordCounter = CountVectorizer(stop_words='english')                       # approach 1: only stop_word
            if stem:
                wordCounter = CountVectorizer(stop_words='english', preprocessor=stemSen)   # approach 2: stop_word & stem
        count_matrix = wordCounter.fit_transform(sentences)
        normalized_matrix = TfidfTransformer().fit_transform(count_matrix)


        # wordCounter = TfidfVectorizer()
        # normalized_matrix = wordCounter.fit_transform(sentences)


        # == similarity among sentences ==
        similarity_graph = normalized_matrix * normalized_matrix.T
        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        scores = nx.pagerank(nx_graph)
        orderedSentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


        if len(orderedSentences) < top:
            top = len(orderedSentences)
        with open(out, "w") as lofile:
            for i in range(0, top):
                lofile.write(orderedSentences[i][1].encode('ascii', 'ignore'))
                lofile.write("\n")


        # print count_matrix


#** Run the code **#
def test():
    solution = TextRank()
    solution.rank()

test()
# print stemSen("having having done did")