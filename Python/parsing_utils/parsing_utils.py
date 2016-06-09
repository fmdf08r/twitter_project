import pandas as pd
from csv import DictReader
from collections import defaultdict, Counter
from nltk.tokenize import RegexpTokenizer
import re
import sys
#import operator

file_path = "../datasets/StanceDataset/"

class ParsingUtils:
    def initialize(self, file_name):
        self.file_name = file_path + file_name
        self.rates = defaultdict(float)
        self.rare_tweets_counter()
        self.threshold = 5.0

    def rare_tweets_counter(self):
        count = Counter()
        self.rates = defaultdict(float)
        with open(self.file_name, 'rU') as f:
            reader = DictReader(f)
            for row in reader:
                s = str(row['Tweet'])
                if '@' in s:
                    ls = s.split(' ')
                    for i in range( len(ls) ):
                        if '@' in ls[i]:
                            count[ ls[i] ] += 1
            f.close()
        total = sum(count.values())
        for k in count.keys():
            self.rates[ k ] = 100 * ( count[k] / float(total) )

    def purify(self, tweet):
        ts = str(tweet).lower()
        ts_l = ts.split(' ')
        for i in range(len(ts_l)):
            if 'http' in ts_l or 'www' in ts_l:
                ts_l[i] = 'HYPERLINK'
            if '@' in ts_l[i]:
                rare = self.rates[ ts_l[i] ] < self.threshold
                if rare:
                    ts_l[i] = 'RARE_USER'
                else: 
                    ts_l[i] = re.sub('@','',ts_l[i])
            if '#' in ts_l[i]:
                ts_l[i] = re.sub('#','',ts_l[i])
            if re.match("^-?[0-9]+$", ts_l[i]):
                ts_l[i] = 'NUM'
        tokenizer = RegexpTokenizer('\w+')
        ts_t = tokenizer.tokenize( ' '.join(ts_l) )
        return ' '.join(ts_t)

    def create_dataFrame(self, target=None):
        data_dict = defaultdict(list)
        names = []
        with open(self.file_name, 'rU') as f:
            reader = DictReader(f)
            names.extend(reader.fieldnames)
            for row in reader:
                for n in names:
                    if n == 'Tweet':
                        tweet = self.purify( row[n] )
                        data_dict[n].append(tweet)
                        continue
                    data_dict[ n ].append( row[ n ] )
            f.close()
            data_df = pd.DataFrame(data_dict, index=None)
            if target:
                cond = data_df[ 'Target' ] == target
                data_df = data_df[ cond ]
        return data_df

sys.stdout.write("Starting Parsing Utility")
myUtil = ParsingUtils()
sys.stdout.write("-> complete\n")
sys.stdout.write("- Initializing Training Set")
myUtil.initialize('train.csv')
sys.stdout.write(" -> complete\n")
sys.stdout.write("- Generating DataFrame")
df = myUtil.create_dataFrame('Hillary Clinton')
sys.stdout.write("-> complete\n")
sys.stdout.write("- Saving to file")
df.to_csv('../datasets/StanceDataset/train_purified.csv',index=False)
sys.stdout.write("-> complete\n")

sys.stdout.write("- Initializing Training Set")
myUtil.initialize('test.csv')
sys.stdout.write(" -> complete\n")
sys.stdout.write("- Generating DataFrame")
df = myUtil.create_dataFrame('Hillary Clinton')
sys.stdout.write("-> complete\n")
sys.stdout.write("- Saving to file")
df.to_csv('../datasets/StanceDataset/test_purified.csv',index=False)
sys.stdout.write("-> complete\n")
