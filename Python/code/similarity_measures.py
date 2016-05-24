from wordAveraging import replace, replace_num
import numpy as np

model_path = "../models/unigram.txt.disney_random.doc2vec.word_model"
stop_words_path = "stop_words.txt"
tweets_path = "../datasets/disney_tweets_purified_" 
wAvgs_path = "../results/disney_word_averages_"
size = 300
filesNbr = 14

class SimilarityMetric:
    def __init__(self):
        self.model = Word2Vec.load(model_path)
        self.stop_words = [line.strip('\n') for line in open(stop_words_path)]
        self.size = size
        self.wordAverages = {}
        for i in range(1,filesNbr + 1):
            tweetsFile = tweets_path + str(i) + ".csv"
            wAvgsFile = wAvgs_path + str(i) + ".csv"
            with open(tweetsFile, 'r') as f1:
                with open(wAvgsFile, 'r') as f2:
                    tw = tweetsFile.readline()
                    tw_vec = wAvgsFile.readline()
                    self.wordAverages[tw] = tw_vec

    def purify(self, tweet):
        tw = str(tweet).lower()
        if 'http' in tw or 'www' in tw:
            tw = replace(tw)
        if '@' in tw:
            tw = replace(tw)
        tw = replace_num(tw)
        return tw

    def tweet2vec(self, tw):
        t = self.purify(tw)
        vec = np.zeros(self.size)
        if tw in self.wordAverages.keys():
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
        tw_vec = self.get_vector_rep(tw)
        k_vec = self.get_vector_rep(keywords)
        return np.dot(tw_vec, k_vec)

    def similarity_2(self, tweet, tweet_vec):
        print "similarity 2"
        tw_vec = self.get_vector_rep(tweet)
        tv_vec = self.get_vector_rep(tweet_vec, isTweet=True)
        return np.dot(tw_vec, tv_vec)

# tests





