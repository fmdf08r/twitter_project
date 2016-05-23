import pandas as pd
from nltk.tokenize import RegexpTokenizer

df1 = pd.read_csv('datasets/disney_tweets_1.csv',header=None)
df2 = pd.read_csv('datasets/disney_tweets_2.csv',header=None)
df3 = pd.read_csv('datasets/disney_tweets_3.csv',header=None)
df4 = pd.read_csv('datasets/disney_tweets_4.csv',header=None)
df5 = pd.read_csv('datasets/disney_tweets_5.csv',header=None)
df6 = pd.read_csv('datasets/disney_tweets_6.csv',header=None)
df7 = pd.read_csv('datasets/disney_tweets_7.csv',header=None)
df8 = pd.read_csv('datasets/disney_tweets_8.csv',header=None)
df9 = pd.read_csv('datasets/disney_tweets_9.csv',header=None)
df10 = pd.read_csv('datasets/disney_tweets_10.csv',header=None)
df11 = pd.read_csv('datasets/disney_tweets_11.csv',header=None)
df12 = pd.read_csv('datasets/disney_tweets_12.csv',header=None)
df13 = pd.read_csv('datasets/disney_tweets_13.csv',header=None)
df14 = pd.read_csv('datasets/disney_tweets_14.csv',header=None)

df = pd.concat(  [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14]  , axis=0)


#df1 = pd.read_csv('datasets/cikm_test_tweets.txt', header=None)
#df2 = pd.read_csv('datasets/cikm_training_tweets.txt', header=None)
#df = pd.concat( [df1, df2], axis=0 )

def replace(s):
    s_lst = s.split(' ')
    for i in range(len(s_lst)):
        if 'http' in s_lst[i] or 'www' in s_lst[i]:
            s_lst[i] = 'HYPERLINK'
        if '@' in s_lst[i]:
            s_lst[i] = 'USER'
    s = ' '.join(s_lst)
    return s

print "duplicate elimination: starting"
duplicates = {}
tweets = df.iloc[:,0].values.tolist()
parsed = []
count = 0
for i in range( len(tweets) ):
    twt = tweets[i].lower()
    if 'http' in twt or 'www' in twt:
        twt = replace(twt)
    if '@' in twt:
        twt = replace(twt)
    tokenizer = RegexpTokenizer('\w+')
    lst = tokenizer.tokenize(twt)
    twt = ' '.join(lst)
    if i % 500000 == 0: print("iteration: " + str(i) )
    if twt in duplicates:
        count += 1
        continue
    else:
        duplicates[twt] = 1
        parsed.append(twt)
print "duplicate elimination: completed"
print "duplicates found: " + str(count) + " out of " + str(len(tweets))
print "saving to file: starting"
with open('datasets/disney_tweets_no_duplicates.txt', 'w') as f:
    for twt in parsed:
        f.write(twt + '\n')
    f.close()
print "saving to file: completed"
