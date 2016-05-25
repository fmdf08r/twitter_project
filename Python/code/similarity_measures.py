from wordAveraging import replace, replace_num
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
import csv

model_path = "../models/unigram.txt.disney_random.doc2vec.word_model"
stop_words_path = "stop_words.txt"
tweets_path = "../datasets/disney_tweets_purified_" 
wAvgs_path = "../results/disney_word_averages_"
size = 300
filesNbr = 1

class SimilarityMetric:
    def __init__(self):
	print "initializing metric:"
        self.model = Word2Vec.load(model_path)
        print "w2v model loaded"
	self.stop_words = [line.strip('\n') for line in open(stop_words_path)]
        self.size = size
        self.wordAverages = {}
	print "loading word averages:"
        for i in range(1,filesNbr + 1):
            print "file number: " + str(i)
	    tweetsFile = tweets_path + str(i) + ".csv"
            wAvgsFile = wAvgs_path + str(i) + ".csv"
	    tweets = []
	    values = []
	    with open(tweetsFile, 'r') as f1: 
		tweets = f1.readlines()
		f1.close()
            with open(wAvgsFile, 'r') as f2: 
		reader = csv.reader(f2)
		for r in reader:
		    values.append( np.array([ float(v) for v in r  ]) )
		f2.close()
	    for j in range(len(tweets)):   
		self.wordAverages[ tweets[j].strip('\n')  ] = values[j]
	print "--> completed"

    def purify(self, tweet):
        tw = str(tweet).lower()
        if 'http' in tw or 'www' in tw:
            tw = replace(tw)
        if '@' in tw:
            tw = replace(tw)
	tokenizer = RegexpTokenizer('\w+')
	lst = tokenizer.tokenize(tw) 
        for j in range(len(lst)):
	    lst[j] = replace_num(lst[j].decode(encoding='utf-8',errors='ignore'))
        return ' '.join(lst)

    def tweet2vec(self, tw):
	#print "original: " + tw
        t = self.purify(tw)
	#print "purified: " + t
	vec = np.zeros(self.size)
        if tw in self.wordAverages.keys():
	    #print "tw is in"
	    vec = self.wordAverages[tw]
        return vec

    def word2vec(self, w):
        w = str(w).lower()
        vec = np.zeros(self.size)
        if w in self.model.vocab.keys():
            vec = self.model[w]
        return vec

    def get_vector_rep(self, x, isTweet=False):
        if isinstance(x, basestring):
            return self.tweet2vec(x)    
        if isinstance(x, list):
            vec = np.zeros(self.size)
            for k in x:
                if isTweet:
                    vec += self.tweet2vec(k)
                else:
                    vec += self.word2vec(k)
            return vec

    def similarity_1(self, tweet, keywords):
        print "similarity 1"
        tw_vec = self.get_vector_rep(tweet)
        k_vec = self.get_vector_rep(keywords)
	return np.dot(tw_vec, k_vec)

    def similarity_2(self, tweet, tweet_vec):
        print "similarity 2"
        tw_vec = self.get_vector_rep(tweet)
        tv_vec = self.get_vector_rep(tweet_vec, isTweet=True)
        return np.dot(tw_vec, tv_vec)

# tests
myMetric = SimilarityMetric()

tweets = pd.read_csv('../datasets/disney_tweets_1.csv',header=None)
tweets = tweets.iloc[0:5].values.tolist()

key_words = ['starwars']
for t in tweets:
    # test similarity 1
    val = myMetric.similarity_1(t[0], key_words)
    print val
