from wordAveraging import replace, replace_num
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import RegexpTokenizer
from math import sqrt
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import sys
import csv

model_path = "../models/unigram.txt.disney_random.doc2vec.word_model"
stop_words_path = "stop_words.txt"
tweets_path = "../datasets/disney_tweets_purified_" 
wAvgs_path = "../results/disney_word_averages_"
size = 300

class SimilarityMetric:
    def __init__(self, filesNbr):
        self.filesNbr = filesNbr
        self.size = size

    def initialize(self):
        sys.stdout.write("Metric initialization\n")
        sys.stdout.write("1 - Word2vec model")
        self.model = Word2Vec.load(model_path)
        sys.stdout.write("...loaded\n")
        sys.stdout.write("2 - Stop words")
        self.stop_words = [line.strip('\n') for line in open(stop_words_path)]
        sys.stdout.write("...loaded\n")
        sys.stdout.write("3 - Word-Averages model: ")
        self.wordAverages = defaultdict()
        for i in range(1, self.filesNbr + 1):
            sys.stdout.write(str(i) + " - ")
            sys.stdout.flush()
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
        sys.stdout.write("loaded\n")

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

    def tweet2vec(self, tw, speed_up = False):
        t = self.purify(tw)
        if speed_up: return self.wordAverages[t]
        vec = np.zeros(self.size)
        if t in self.wordAverages:
            vec = self.wordAverages[t]
        return vec

    def word2vec(self, w):
        w = str(w).lower()
        vec = np.zeros(self.size)
        if w in self.model.vocab:
            vec = self.model[w]
        return vec

    def get_vector_rep(self, x, isTweet=False,speed_up=False):
        if isinstance(x, basestring):
            return self.tweet2vec(x,speed_up)    
        if isinstance(x, list):
            vec = np.zeros(self.size)
            for k in x:
                if isTweet:
                    vec += self.tweet2vec(k)
                else:
                    vec += self.word2vec(k)
            return vec / len(x)
        if isinstance(x, dict):
            vec = np.zeros(self.size)
            for k, v in x.iteritems():
                if v == '+':
                    if isTweet:
                        vec += self.tweet2vec(k)
                    else:
                        vec += self.word2vec(k)
                if v == '-':
                    if isTweet:
                        vec -= self.tweet2vec(k)
                    else:
                        vec -= self.word2vec(k)
            return vec / len( x.keys() )

    def similarity_1(self, tweet, keywords):
        tw_vec = self.get_vector_rep(tweet)
        k_vec = self.get_vector_rep(keywords)
        dot = np.dot(tw_vec, k_vec)
        sqt = sqrt( np.sum( np.multiply( tw_vec, tw_vec ) ) ) * sqrt( np.sum( np.multiply( k_vec, k_vec ) ) )
        if dot == 0.0: return 0.0
        return dot / sqt
    
    def sim_1(self, tweet):
        return self.similarity_1(tweet, self.keywords)

    def similarity_2(self, tweet, tweet_vec):
        tw_vec = self.get_vector_rep(tweet)
        tv_vec = self.get_vector_rep(tweet_vec, isTweet=True)
        dot = np.dot(tw_vec, tv_vec)
        sqt = sqrt( np.sum( np.multiply( tw_vec, tw_vec ) ) ) * sqrt( np.sum( np.multiply( tv_vec, tv_vec ) ) )
        if dot == 0.0: return 0.0
        return dot / sqt

    def get_max(self, keywords, metric):
        max_value = - 10.0
        max_tweet = 'no_tweet'
        k_vec = np.zeros(self.size)
        if metric == 1:
            k_vec = self.get_vector_rep(keywords)
        if metric == 2:
            k_vec = self.get_vector_rep(keywords, isTweet=True)
        for k, v in self.wordAverages.iteritems():
            dot = np.dot(v, k_vec)
            sqt = sqrt( np.sum( np.multiply( v, v ) ) ) * sqrt( np.sum( np.multiply( k_vec, k_vec ) ) )
            val = -10.0
            if dot == 0.0: val = 0.0
            else: val = dot / sqt
            if val > max_value:
                max_value = val
                max_tweet = k
        return (max_tweet, max_value)

    def sort(self, keywords, descending):
        self.keywords = keywords
        tweets_sorted = sorted(self.wordAverages, reverse=descending, key=self.sim_1)
        return tweets_sorted

    def get_k_max(self, keywords, k):
        sorted_tweets = self.sort(keywords, True)
        output = {}
        for i in range(k):
            s = sorted_tweets[ i ]
            output[ s ] = self.similarity_1(s, keywords) 
        return output

# tests
myMetric = SimilarityMetric(1)
myMetric.initialize()

df = pd.read_csv('../datasets/disney_tweets_1.csv',header=None)
keywords_1 = ['starwars', 'luke', 'leia', 'disney']
tweets = df.iloc[45:50].values.tolist()
keywords_2 = {'starwars':'+', 'home':'-', 'chewie':'+'}

import datetime
t1_start = datetime.datetime.now()
(t1, m1) = myMetric.get_max(keywords_1, 1)
t1_end = datetime.datetime.now()
sys.stdout.write("max 1 =" + str(m1) + " for tweet: " + t1 + "\n" ) 
sys.stdout.write("calculating max 1 took: " + str( t1_end - t1_start ) + "\n" )

t2_start = datetime.datetime.now()
(t2, m2) = myMetric.get_max(tweets, 2)
t2_end = datetime.datetime.now()
sys.stdout.write("max 2 =" + str(m2) + " for tweet: " +  t2 + "\n" )
sys.stdout.write("calculating max 2 took: " + str( t2_end - t2_start ) + "\n" )

t3_start = datetime.datetime.now()
(t3, m3) = myMetric.get_max(keywords_2, 1)
t3_end = datetime.datetime.now()
sys.stdout.write("max 3 =" + str(m3) + " for tweet: " +  t3 + "\n" )
sys.stdout.write("calculating max 3 took: " + str( t3_end - t3_start ) + "\n" )

t4_start = datetime.datetime.now()
sorted_tweets = myMetric.get_k_max(keywords_2, 5)
t4_end = datetime.datetime.now()
sys.stdout.write("best 5 tweets: \n")
for k, v in sorted_tweets.iteritems():
    sys.stdout.write(str(k) + "\n" )
    sys.stdout.write("val = " + str(v) + '\n')
sys.stdout.write("calculating 5 maxes  took: " + str( t4_end - t4_start ) + "\n" )

