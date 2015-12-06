__author__ = 'zhangnan'

#   NLP
#   appendTag.py

#Created by zhangnan on 12/5/15.

from nltk.tokenize import word_tokenize
import nltk

def appendTag(sentences):
    tags = []
    for line in sentences:
        text = word_tokenize(line)
        tags.append(nltk.pos_tag(text))

    fullsentences = [" ".join(['/'.join(token)
                               for token in sentence])
                     for sentence in tags]

    return fullsentences

# s = ['I love you', 'i hate you']
# print appendTag(s)