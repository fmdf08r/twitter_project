import pandas as pd
import numpy as np
import logging
import sys
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases
from collections import defaultdict

files_path = "../datasets/StanceDataset/"
extension = ".csv"

class W2V_Generator: 
    def __init__(self, name):
        self.name = name
        self.wordAverages = defaultdict(list)
        self.df = pd.read_csv(files_path + name + extension, header=0)
        self.w2v_size = 100

    def calcW2Vrep(self, log):
        tweets = self.df.loc[:,'Tweet']
        t_list = []
        for t in tweets:
            tl = t.split(' ')
            for i in range( len( tl ) ):
                tl[i] = tl[i].decode(encoding='utf-8', errors='ignore')
            t_list.append( tl )
        unigrams = Phrases(t_list, threshold=5.0, min_count=5)
        if log:
            logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
            logging.info( "running %s" % " ".join(sys.argv) )
    
        tweets_unigrams = unigrams[ t_list ]
        self.model = Word2Vec( tweets_unigrams, min_count=5, workers=12, size=self.w2v_size, window=5, negative=5, iter=50 )

    def save_w2v_model(self):
        self.model.save(files_path + self.name + ".doc2vec.word_model")
        self.model.save_word2vec_format(files_path + self.name + ".doc2vec.word_vec")

    def save_wordAvgs(self):
        with open(files_path + self.name + ".wAvgs.tweets" + extension, 'wb') as tw:
            tw.write("Tweet, Stance, Sentiment\n")
            with open(files_path + self.name + ".wAvgs.values" + extension, 'wb') as tv:
                for t, v in self.wordAverages.iteritems():
                    try:
                        row = self.df.loc[ self.df['Tweet'] == t ]
                        stance = str( row['Stance'].values[0] )
                        sentiment = str( row['Sentiment'].values[0] )
                        tw.write(str(t) + "," + stance + "," + sentiment + "\n")                
                        for i in range(v.size):
                            tv.write( str(v[i]) )
                            if i < v.size - 1: tv.write( "," )
                            else: tv.write("\n")
                    except KeyError:
                        sys.stdout.write(str(idx) + ': ' + t)
                        sys.stdout.write(" || NOT FOUND in the dataframe\n")
                tv.close()
            tw.close()

    def convert_labels(self, task):
        categories = {}
        if task == 'Stance':
            categories[ 'AGAINST' ] = -1
            categories[ 'FAVOR' ] = 1
            categories[ 'NONE' ] = 0
        
        if task == 'Sentiment':
            categories[ 'neg' ] = -1
            categories[ 'pos' ] = 1
            categories[ 'other' ] = 0

        for idx, row in self.df.iterrows():
            row[ task ] = categories[ row[ task ] ]
        
    def calcWordAverages(self):
        stop_words = [line.strip('\n') for line in open('../datasets/stop_words.txt','r')]
        self.wordAvgs = defaultdict(np.ndarray)
        for i, tweet in self.df['Tweet'].iteritems():
            size = len( tweet )
            tw_list = tweet.split(' ')
            avg = np.zeros(self.w2v_size)
            for word in tw_list:
                if word in stop_words: continue
                if str(word) in self.model.vocab:
                    avg += self.model[ word ]
            self.wordAverages[ tweet ] = avg / float (size)    

######################################
######################################
######################################

# TRAINING SET
sys.stdout.write("1 - Training SET: set up W2V utilities: \n")
trainW2V = W2V_Generator('train_purified')
sys.stdout.write("-- Converting Sentiment / Stance to int")
trainW2V.convert_labels('Sentiment')
trainW2V.convert_labels('Stance')
sys.stdout.write(" -> complete\n")
sys.stdout.write("-- Generating / Saving W2V representation")
trainW2V.calcW2Vrep(False)
trainW2V.save_w2v_model()
sys.stdout.write(" -> complete\n")
sys.stdout.write("-- Calculating / Saving Tweet averages")
trainW2V.calcWordAverages()
trainW2V.save_wordAvgs()
sys.stdout.write(" -> complete\n")

# TEST SET
sys.stdout.write("1 - Test SET: set up W2V utilities: \n")
testW2V = W2V_Generator('test_purified')
sys.stdout.write("-- Converting Sentiment / Stance to int")
testW2V.convert_labels('Sentiment')
testW2V.convert_labels('Stance')
sys.stdout.write(" -> complete\n")
sys.stdout.write("-- Generating / Saving W2V representation")
testW2V.calcW2Vrep(False)
testW2V.save_w2v_model()
sys.stdout.write(" -> complete\n")
sys.stdout.write("-- Calculating / Saving Tweet averages")
testW2V.calcWordAverages()
testW2V.save_wordAvgs()
sys.stdout.write(" -> complete\n")
