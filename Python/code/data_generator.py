from disney_tweets import create_df, clean_df
import pandas as pd

#######################
## generate data set ##
#######################
disney_df_1 = create_df('datasets/disney_tweets_1')
disney_df_2 = create_df('datasets/disney_tweets_2')
disney_df_3 = create_df('datasets/disney_tweets_3')
#disney_df_4 = create_df('datasets/disney_tweets_4')
#disney_df_5 = create_df('datasets/disney_tweets_5')
#disney_df_6 = create_df('datasets/disney_tweets_6')
#disney_df_7 = create_df('datasets/disney_tweets_7')
#disney_df_8 = create_df('datasets/disney_tweets_8')
#disney_df_9 = create_df('datasets/disney_tweets_9')
#disney_df_10 = create_df('datasets/disney_tweets_10')
#disney_df_11 = create_df('datasets/disney_tweets_11')
#disney_df_12 = create_df('datasets/disney_tweets_12')
#disney_df_13 = create_df('datasets/disney_tweets_13')
#disney_df_14 = create_df('datasets/disney_tweets_14')

general_1 = create_df('datasets/general_1')
general_2 = create_df('datasets/general_2')
general_3 = create_df('datasets/general_3')

#disney = [disney_df_1, disney_df_2, disney_df_3, disney_df_4, disney_df_5, disney_df_6, disney_df_7, disney_df_8, disney_df_9, disney_df_10, disney_df_11, disney_df_12, disney_df_13, disney_df_14]
disney = [disney_df_1, disney_df_2, disney_df_3, general_1, general_2, general_3]

df = pd.concat(disney, axis=0)
disney_clean = clean_df(df, 'disney_random_tweets')

