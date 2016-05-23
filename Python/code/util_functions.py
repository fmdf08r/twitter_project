from nltk.tokenize import RegexpTokenizer
import pandas as pd
import gensim, logging
import numpy as np
import random
import tsne

# create a dataframe by reading a specific file of tweets
def create_df(name):
    filename = name + '.csv'
    df = pd.read_csv(filename, header=None, delimiter='\t',error_bad_lines=False,dtype=str)
    df.columns=['tweet']
    df.dropna(axis=0,how='any',inplace=True)        
    df.info()
    return df

# filter tweets to eliminate hyperlinks and usernames
def replace(s):
    s_lst = s.split(' ')
    for i in range(len(s_lst)):
        if 'http' in s_lst[i] or 'www' in s_lst[i]:
            s_lst[i] = 'HYPERLINK'
        if '@' in s_lst[i]:
            s_lst[i] = 'USER'
    s = ' '.join(s_lst)
    return s

# format the tweets within a dataframe to eliminate keywords and punctuation
def clean_df(df,name):
    df_iterator = df.iterrows()
    idx = 1
    str_list = []
    lst_list = []
    k = 1000000
    step = 1000000
    for i, row in df_iterator:
        s = row['tweet'].lower()
        if 'http' in s or 'www' in s:
            s = replace(s)
        if '@' in s:
            s = replace(s)
        tokenizer = RegexpTokenizer('\w+')
        lst = tokenizer.tokenize(s)
        if i > k:
            print 'conversion completed up to: ' + str(k) + ' lines'
            k += step
        s = ' '.join(lst)
        if s == '""' or len(s) < 3:
            continue
        # add to a list
        lst_list.append(lst)
        str_list.append(s)
        idx += 1    
    df_str = pd.DataFrame({'tweet': str_list})
    df_str.to_csv(name + '_tweets.txt',header=False,index=False)
    return lst_list

# filters a dataframe to produce a unicode object
def read_df(df):
    df_iter = df.iterrows()
    str2lst = []
    k = 200000
    step = 200000
    for i, row in df_iter:
        #print row[0]
        s = str(row[0])
        tokenizer = RegexpTokenizer('\w+')        
        lst = tokenizer.tokenize(s)
        for e in lst:
            e.decode(encoding='utf-8', errors='ignore')
        if i > k:
            print("iteration: " + str(k))
            k += step
        str2lst.append(lst)
    return str2lst 

# implements weighted sampling of an element
def weighted_choice_sub(weights):
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

# implements sampling N integers using weights
def get_indices(weights, X, N):

    indices = []
    size = 0
    while size != N:
        idx = weighted_choice_sub(weights)
        #print idx
        if idx not in indices:
            too_small = False
            for x in X[idx]: 
                if abs(x) < 0.0001: 
                    too_small = True
                    break
            if too_small: continue
            indices.append(idx)
            size += 1
    return indices

# create two files: one for the model and one for the labels
def make_score_files(model, filelabel):
    with open(filelabel + '_scores.csv', 'w') as scorefile:
        with open(filelabel + '_words.csv', 'w') as wordfile:
            count_scores = 0
            count_words = 0
            for i in range(len(model.index2word)):
                word = model.index2word[i]
                if i == 80:
                    continue
                try:
                    wordfile.write(word + '\n')
                    score = model[word]
                    scores = [str(x) for x in score]
                    scorefile.write('\t'.join(scores) + '\n')
                    #print("word: " + word)
                    count_scores += 1
                    count_words += 1
                except:
                    print "Not found: " + str(i) + ": " + word
                    continue
            print("counts: scores = " + str(count_scores) + "; words = " + str(count_words))     
            #scorefile.write('\n')
            #wordfile.write('\n') 
