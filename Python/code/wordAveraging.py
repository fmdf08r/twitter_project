import pandas as pd
from multiprocessing import Process
import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import RegexpTokenizer
import re

def replace(s):
    s_lst = s.split(' ')
    for i in range(len(s_lst)):
        if 'http' in s_lst[i] or 'www' in s_lst[i]:
            s_lst[i] = 'HYPERLINK'
        if '@' in s_lst[i]:
            s_lst[i] = 'USER'
    s = ' '.join(s_lst)
    return s

def replace_num(x):
    if re.match("^-?[0-9]+$",x):
        return "NUM"
    else:
        return x

def read_df(df):
    df_iter = df.iterrows()
    str2lst = []
    for i, row in df_iter:
        s = str(row[0]).lower()
        if 'http' in s or 'www' in s:
            s = replace(s)
        if '@' in s:
            s = replace(s)
        tokenizer = RegexpTokenizer('\w+')
        lst = tokenizer.tokenize(s)
        for j in range(len(lst)):
            lst[j] = replace_num(lst[j].decode(encoding='utf-8',errors='ignore'))
        if i % 200000 == 0: print("reading line: " + str(i))
        str2lst.append(lst)
    return str2lst

def wordCounter(nbr, fileName, w2v_size):
    print("running: " + str(nbr))
    model = Word2Vec.load('../models/unigram.txt.disney_random.doc2vec.word_model')
    stop_words = [line.strip('\n') for line in open("stop_words.txt",'r')]
    avgDict = {}
    df = pd.read_csv(fileName, header=None)
    tweets = read_df(df)
    print(str(nbr) + " starting computation")
    total = len(tweets)
    for i in range(total):
        if i % 100000 == 0: print("t_" + str(nbr) + ": iteration: " + str(i))
        length = len(tweets[i])
        avg = np.zeros(w2v_size)
        ls = ' '.join(tweets[i])
        for word in tweets[i]:
            if word in stop_words: continue
            if str(word) in model.vocab.keys():
                avg += model[word] 
        avgDict[ls] = avg / float(length)
    # print everything
    print("t_" + str(nbr)+ ": printing to file")
    saveFile = "../results/disney_word_averages_" + str(nbr) + ".csv"
    tweetsFile = "../datasets/disney_tweets_purified_" + str(nbr) + ".csv"
    with open(saveFile, 'wb') as f1:
        with open(tweetsFile, 'wb') as f2:
            for k, v in avgDict.iteritems():
                f2.write(str(k) + "\n")
	        for i in range( len(v) ):
                    f1.write( str(v[i] ) )
                    if i < len(v) - 1:
                        f1.write(", ")
                    else: 
                        f1.write("\n")
            f2.close()
	f1.close()
    print("t_" + str(nbr)+ ": completed")

'''
###############
## MAIN FILE ##
###############
filesNbr = 14
print("Main Thread: word count starting: ")
wcts = []
avgDicts = []
for i in range(1, filesNbr+1):
    fileName = '../datasets/disney_tweets_' + str(i) + '.csv'
    w2v_size = 300
    p = Process(target=wordCounter, args=(i, fileName, w2v_size))    
    p.start()
    wcts.append(p)
    #wct = WordCounterThread(i, fileName)
    #wct.start()
    #wcts.append(wct)

for t in wcts: 
    t.join()
print("Main Thread: word count completed")
'''
