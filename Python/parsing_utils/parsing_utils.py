import pandas as pd
from csv import DictReader
from collections import DefaultDict, Counter
from nltk.tokenize import RegexpTokenizer
import re

file_path = "../datasets/StanceDataset/"
class ParsingUtils:
    def initialize(self, file_name):
        self.file_name = file_path + file_name
        self.count = Counter()
        self.rare_tweets_counter()
        self.threshold = 5.0

    def rare_tweets_counter(self):
        with open(self.file_name, 'rU') as f:
            reader = DictReader(f)
            for row in reader:
                s = str(row['Tweet'])
                if '@' in s:
                    ls = s.split(' ')
                    for i in range( len(ls) ):
                        if '@' in ls[i]:
                            self.count[ ls[i] ] += 1
            f.close()
        total = sum(self.count.values())
        rates = DefaultDict(float)
        for k in self.count.keys():
            rates[ k ] = 100 * ( self.count[k] / float(total) )
        return rates

    def purify(self, tweet):
        ts = str(tweet).lower()
        ts_l = ts.split(' ')
        for i in range(len(ts_l)):
            if 'http' in ts_l or 'www' in ts_l:
                ts_l[i] = 'HYPERLINK'
            if '@' in ts_l[i]:
                rare = self.count[ ts_l[i] ] < self.threshold
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

    def create_dataFrame(self):
        data_dict = DefaultDict(list)
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
            data_df = pd.DataFrame(data_dict,)
        return data_df
