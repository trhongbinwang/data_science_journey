# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:55:37 2016

pandas summary


"""
import pickle
import pandas as pd

# used in notebook
import matplotlib.pyplot as plt
%matplotlib inline

#.iloc() -- position index; .loc() -- label index. 

# general info
df.head(), df.info()
# read excel data from a folder
    from os import listdir
    from os.path import isfile, join
    # windows change from \ to /
    mypath = 'C://Documents/data//'
    fileNames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    df_all = pd.DataFrame() # empty df
    for fileName in fileNames:
        fileName = join(mypath, fileName)
        df = pd.read_excel(fileName)
        print(df.shape)
        df_all = df_all.append(df)
        print(df_all.shape)
    print(df_all.info())
# write into a excel file
    writer = ExcelWriter('historical.xlsx')
    df.to_excel(writer, 'fx')
    writer.save()  

# dump or load pickle
    with open('historical.pkl', 'wb') as f:
        pickle.dump(df_all, f)
    df_loaded = pickle.load(open('historical.pkl', 'rb'))

# pd select columns
df_all = df_all[['level', 'name']]
# pd filtering
df = df[df['st'] != '-']
df = df[df['tr'] > 60]
# convert time from string to datetime
df['time']= pd.to_datetime(df['time'], infer_datetime_format=True)
# filtering the time
df_past = df[df['happen_time'] < datetime.datetime(2016,6,1,0,0,0)]
# pd sort by a column value
df = df.sort_values(by='st')
# format print
print('number of action = {0}'.format(len(df)))
# iterate dataframe
    for i in range(1,len(df)):
        if ((df.iloc[i]['name'] == df.iloc[i-1]['name']) # two index access maynot can change it
# generate new collumn from existing columns
    df['tr'] = (df['st1'] - df['st']).apply(lambda x: x.total_seconds())
    df['source'] = df['s1'] + ' / ' + df['s2']
    
# drop na and duplicates
df = df.drop_duplicates(['name', 'tv'])
df = df.dropna(subset = ['name'])

# convert pd into numpy array
pd.values
# another way to iterate pd row by row
    for row in df.itertuples(index=False):
        print(row) # row is a list
# group data using column
    grouped = df.groupby('source')
    for name, group in grouped:
# set value to an element in df
df.iloc[i, df.columns.get_loc('ro')] = 'de'
# drop columns from df
df = df.drop(['level', 'tr'], axis=1)

# pd concat or append
df = pd.concat([df, df_latest], axis=0) 

# create pandas df from list of tuple, also can from a list or a dict
    name_fre = [(name, len(group)) for name, group in grouped]
    name_fre_df = pd.DataFrame(name_fre, columns=['name', 'Freq'])
# double condition selection
    small_alarms = df[(name_fre_df['Frequency']>10) & (name_fre_df['Frequency']<100)]['name'].values.tolist()
# only select value from a list
df_large = df[df['name'].isin(large)]
# pd one hot encoding categorical data, better one use sklearn or write your own
hour_dummies = pd.get_dummies(ml_df['hour'],prefix='hour')
# apply func to a column
ip_head = ip_head.apply(lambda x: x.split('.')[0])


# add a column in pd
data['price'] = 0
# delete a column
del data['price']
# rename a column
data = data.rename(columns={'NAME':'PLANET'})

# The column names, as an Index object
print( df.columns )
# convert to list
list(df.columns)

# view pd's column in two styles
df.beer_style# 1. attribute style
df['beer_style'] # 2. dict style

# string operation and selection
boolean_mask = df.beer_style.str.contains('[A|a]merican')
df.beer_style[boolean_mask]

# Statistical Operations through Indexing
beer_ids = df.beer_id.value_counts(sort=True) # count frequency of each id

# iloc access
# row, column
df.iloc[[2,5,10],0:3]


# lambda funciton
lambda x: x= 0

# 3 steps 
grouped = data.groupby(lambda x: x.year) #1. group
grouped.get_group(1997) #2.  split

gr = df.groupby('beer_style') # gr is a groupby object
gr.agg('mean') # get some statistics from each group
# only a subset
review_columns = ['abv','review_overall','review_appearance',
                  'review_palate','review_taste']
gr[review_columns].agg('mean')

gr['review_aroma'].agg([np.mean, np.std, 'count']) # multiple aggregation










