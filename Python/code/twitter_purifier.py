import pandas as pd
from multiprocessing import Process
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
    if re.match("^-?[0-9]+$", x):
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

def tweet_printer(nbr, fileName, outputFileName):
    print("running: " + str(nbr))
    df = pd.read_csv(fileName, header=None)
    tweets = read_df(df)
    print(str(nbr) + " starting printing")
    total = len(tweets)
    with open(outputFileName, 'wb') as f:
        for i in range(total):
            if i % 100000 == 0 : print("t_" + str(nbr) + ": iteration: " + str(i))
            ls = ' '.join(tweets[i])
            f.write(ls + "\n")
        f.close()
    print("t_" + str(nbr) + ": completed")

###############
## MAIN FILE ##
###############
filesNbr = 14
print("Main Thread: word count starting: ")
wcts = []
avgDicts = []
for i in range(1, filesNbr+1):
    fileName = 'datasets/disney_tweets_' + str(i) + '.csv'
    outputName = 'datasets/disney_tweets_purified_' + str(i) + '.csv'
    p = Process(target=tweet_printer, args=(i, fileName, outputName))
    p.start()
    wcts.append(p)
    #wct = WordCounterThread(i, fileName)
    #wct.start()
    #wcts.append(wct)

for t in wcts:
    t.join()
print("Main Thread: word count completed")
