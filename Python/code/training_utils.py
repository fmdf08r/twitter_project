from nltk.tokenize import RegexpTokenizer
import logging, gensim
import pandas as pd
import re

def replace_num(x):
    if re.match("^-?[0-9]+$", x):
        return "NUM"
    else:
        return x

# filters a dataframe to produce a unicode object
def read_df(df):
    df_iter = df.iterrows()
    str2lst = []
    step = 200000
    for i, row in df_iter:
        #print row[0]
        s = str( row[0] )
        tokenizer = RegexpTokenizer('\w+')        
        lst = tokenizer.tokenize(s)
        for j in range(len(lst)): 
	    lst[j] = replace_num(lst[j].decode(encoding='utf-8', errors='ignore'))
        if i % step == 0:print("iteration: " + str(i))
        str2lst.append(lst)
    return str2lst 

df_1 = pd.read_csv('datasets/disney_tweets_no_duplicates.txt',error_bad_lines=False,header=None)
df_2 = pd.read_csv('datasets/random_tweets_no_duplicates.txt',error_bad_lines=False,header=None)
total = [df_1, df_2]
#total = [df_1.iloc[0:30000], df_2.iloc[0:30000]]
df = pd.concat(total,axis=0)
tweets = read_df(df)

step = 200000
l = []
for i in range(len(tweets)):
    lst = tweets[i]
    if i % step == 0:
        print("iteration: " + str(i))
    #lst_u = [replace_num( str(s) ).decode(encoding='ascii', errors='replace') for s in lst]
    l.append(lst)

#bigram = gensim.models.phrases.Phrases.load('disney_random_bigrams.txt')
#trigram = gensim.models.phrases.Phrases.load('disney_random_trigrams.txt')

unigram = gensim.models.phrases.Phrases(l, threshold=15, min_count=100)
unigram.save("mixed_unigrams.txt")

docs = unigram[l]

file = open("unigram.txt", "wb")

for d in docs:
  for s in d:
    file.write("%s " % s.encode('utf-8') )
  file.write("\n")

file.close()

"""
print("5) Starting Training: ")
model = gensim.models.word2vec.Word2Vec(min_count=freq, size=size_NN, workers=nbr_threads)  # use fixed learning rate min_alpha=0.025 alpha=0.025
start = time.time()
model.build_vocab(trigram[bigram[l]])
end = time.time()
print("   - Building vocabulary completed in: " + str(end - start))
for epoch in range(10):
    print('     - starting epoch: ' + str(epoch))
    start = time.time()
    random.shuffle(l)
    end = time.time()
    print("     - shuffling completed in: " + str(end - start))
    start = time.time()
    model.train(trigram[bigram[l]])
    #model.save('disney_' + str(epoch) + '.txt')
    end = time.time()
    print("     - training completed in: " + str(end - start))
    #model.alpha -= 0.002  # decrease the learning rate
    #model.min_alpha = model.alpha  # fix the learning rate, no decay
"""

