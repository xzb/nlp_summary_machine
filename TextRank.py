# -*- coding: utf-8 -*-
__author__ = 'xiezebin'

import networkx as nx
import numpy as np

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class Tokenize(object):
    def __init__(self):
        with open("doc.txt", "r") as lofile:
            document = lofile.read().decode('utf-8').replace('\n', ' ').replace('?"', '? "').replace('!"', '! "').replace('."', '. "')
            # document = lofile.read().replace('\n', ' ')

        # document = ' '.join(document.strip().split('\n'))

        # == sentences tokenize ==
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
        sentence_tokenizer = PunktSentenceTokenizer(punkt_param)

        # document = document.replace('?"', '? "').replace('!"', '! "').replace('."', '. "')
        sentences = sentence_tokenizer.tokenize(document)


        wordCounter = CountVectorizer()
        count_matrix = wordCounter.fit_transform(sentences)
        # bow_matrix.toarray()
        normalized_matrix = TfidfTransformer().fit_transform(count_matrix)

        similarity_graph = normalized_matrix * normalized_matrix.T

        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        scores = nx.pagerank(nx_graph)
        orderedSentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


        with open("doc_textrank.txt", "w") as lofile:
            for i in range(0, 3):
                lofile.write(orderedSentences[i][1].encode('ascii', 'ignore'))


        print ""

#** Run the code **#
def test():
    solution = Tokenize()

test()
