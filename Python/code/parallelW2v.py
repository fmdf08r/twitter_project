#!/usr/bin/env python

"""

"""

import logging
import sys
import os
import numpy
from gensim.models.word2vec import Word2Vec
from random import shuffle

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

input_file = 'bigram.txt'

# read files
docs = []
with open(input_file) as f:
    contents = f.readlines()

for i in xrange(0, len(contents)-1):
    line = contents[i]
    arr = line.split()        
    docs.append(arr)

model = Word2Vec(docs, min_count=100, workers=24, size=300, window=5, negative=5)

for epoch in range(8):
    shuffle(docs)
    logging.info("training epoch %s" % epoch)
    model.train(docs)

model.save(input_file + '.doc2vec.word_model')
model.save_word2vec_format(input_file + '.doc2vec.word_vec')
