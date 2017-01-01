# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:55:37 2016

pandas summary

http://pandas.pydata.org/pandas-docs/stable/basics.html

can search this page to find howto

0 or ‘index’ for row-wise, 1 or ‘columns’ for column-wise

"""
import pickle
import pandas as pd
import numpy as np

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
        print(row) # row is a namedtuple
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


#
#Attributes and the raw ndarray(s)
# shape; columns and index
df.columns = [x.lower() for x in df.columns]

# Flexible binary operations
row = df.ix[1]
column = df['two']
df.sub(row, axis='columns')

df + df2
df.add(df2, fill_value=0)
df.gt(df2)

# Descriptive statistics; DataFrame: “index” (axis=0, default), “columns” (axis=1)
df.mean(0)
df.sum(0, skipna=False)
# normalize column
ts_stand = (df - df.mean()) / df.std()
# normalize each row
xs_stand = df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)

# index of min and max
df1.idxmin(axis=0)
df1.idxmax(axis=1)

# Value counts (histogramming)
s = pd.Series(np.random.randint(0, 7, size=50))
s.value_counts()

# Row or Column-wise Function Application
df.apply(np.mean)
df.apply(np.mean, axis=1)
# can pass additional parameters to function

# Applying elementwise Python functions
# applymap() for df; map() for Series (column); func takes a single value and output a single value
f = lambda x: len(str(x))
df4['one'].map(f)
df4.applymap(f)

# iteration
for i in object:
# Series, value
# dataframe, column name
for col in df:
    print(col)
# itertuples is a lot faster than iterrows
# itertuples return each row as a namedtuple
# a namedtuple is as row = (Index=0, a=1, b='a'). can be access as: row.a
    
# .dt  for Series
s = pd.Series(pd.date_range('20130101 09:10:12', periods=4))
s.dt.day
s.dt.hour
s.dt.second
s[s.dt.day==2]
s.dt.strftime('%Y/%m/%d') # convert datatime to string

# pd.date_range to generate data range

# Vectorized string methods - Series; exclude missing/NA values automatically
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()
# clean up the columns; chaining because all method return a Series
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Sorting
#sort by index
unsorted_df.sort_index()
# sort by value
df1 = pd.DataFrame({'one':[2,1,1,1],'two':[1,3,2,4],'three':[5,4,3,2]})
df1[['one', 'two', 'three']].sort_values(by=['one','two'])

# smallest / largest values
s = pd.Series(np.random.permutation(10))
s.nsmallest(3)
df = pd.DataFrame({'a': [-2, -1, 1, 10, 8, 11, -1],
                   'b': list('abdceff')})
df.nlargest(3, 'a')
# series.nlargest
s = pd.Series(np.randam.rand(100))
s.largest(10)

# dtypes of each columns
dft.dtypes

#astype
dft = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6], 'c': [7, 8, 9]})
dft[['a','b']] = dft[['a','b']].astype(np.uint8)

#pd.to_datetime()
#to_numeric() 
df.apply(pd.to_datetime)


# Method chaining; since each method return a df. 
def read(fp):
    df = (pd.read_csv(fp)
            .rename(columns=str.lower)
            .drop('unnamed: 36', axis=1)
            .pipe(extract_city_name)
            .pipe(time_to_datetime, ['dep_time', 'arr_time', 'crs_arr_time', 'crs_dep_time'])
            .assign(fl_date=lambda x: pd.to_datetime(x['fl_date']),
                    dest=lambda x: pd.Categorical(x['dest']),
                    origin=lambda x: pd.Categorical(x['origin']),
                    tail_num=lambda x: pd.Categorical(x['tail_num']),
                    unique_carrier=lambda x: pd.Categorical(x['unique_carrier']),
                    cancellation_code=lambda x: pd.Categorical(x['cancellation_code'])))
    return df
# pipe your own function
>>> (df.pipe(h)
...    .pipe(g, arg1=a)
...    .pipe(f, arg2=b, arg3=c)
... )

























