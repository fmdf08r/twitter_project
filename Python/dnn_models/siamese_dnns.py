from keras.models import Model
from keras.layers import Input, Dense
from collections import defaultdict
import sys
import pandas as pd

filepath = '../datasets/StanceDataset/'

def cartesian_product(tw, vs, target):
    vals = defaultdict(list)
    stance = defaultdict(int)
    sentiment = defaultdict(int)
    for idx, r in tw.iterrows():
        tweet = r[ 'Tweet' ]
        vals[ tweet ].extend( vs.iloc[idx,:].values.tolist() )
        stance[ tweet ] = int( r[ 'Stance' ] )
        sentiment[ tweet ] = int( r[ 'Sentiment' ] )
    dataset = defaultdict(list)
    names = []
    for i in range(2 * vs.shape[1]) : names.append( str(i) )
    i = 0
    for i1, r1 in tw.iterrows():
        for i2, r2 in tw.iterrows():
            #####################
            if i % 10000 == 0: 
                sys.stdout.write('iteration: ' + str(i) + '\n')
            i += 1
            #####################
            for n in names: 
                values = []
                el = 0.0
                if int(n) < vs.shape[1]: 
                    values.extend( vals[ r1['Tweet'] ] )
                    el = values[ int(n) ]
                else: 
                    values.extend( vals[ r2['Tweet'] ] )
                    el = values[ int(n) - vs.shape[1] ]
                dataset[ n ].append( el )

            st1 = stance[ r1['Tweet'] ]
            st2 = stance[ r2['Tweet'] ]
            se1 = sentiment[ r1['Tweet'] ]
            se2 = sentiment[ r2['Tweet'] ]
            if target == 0:
                dataset['Stance'].append( int(st1 == st2) )
            elif target == 1:
                dataset['Sentiment'].append( int(se1 == se2))
            else:
                dataset['Stance'].append( int(st1 == st2) ) 
                dataset['Sentiment'].append( int(se1 == se2) )
    cols = names
    if target == 0: cols.append('Stance')
    elif target == 1: cols.append('Sentiment')
    else: cols.extend(['Stance','Sentiment'])
    df = pd.DataFrame(dataset,columns=cols)  

    return df

tw1 = pd.read_csv(filepath + 'train_purified.wAvgs.tweets.csv',header=0)
tw2 = pd.read_csv(filepath + 'test_purified.wAvgs.tweets.csv',header=0)
tweets = pd.concat([tw1, tw2], axis=0)

l1 = pd.read_csv(filepath + 'train_purified.wAvgs.values.csv',header=0)
l2 = pd.read_csv(filepath + 'test_purified.wAvgs.values.csv',header=0)
values = pd.concat([l1, l2], axis=0)

# calculate cartesian product
df = cartesian_product(tweets, values, 0) # 0: Stance ; 1: Sentiment; 2: Stance & Sentiment
df.to_csv('data.csv',header=True,index=False)
