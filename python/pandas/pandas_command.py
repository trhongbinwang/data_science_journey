# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:07:14 2016

pandas command collection


"""

import pandas as pd

import numpy as np

# create df from dict
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])


capitalizer = lambda x: x.upper()
# apply() can apply a function along any axis of the dataframe
df['name'].apply(capitalizer)
# map() applies an operation over each element of a series
df['name'].map(capitalizer)
#applymap() applies a function to every single element in the entire dataframe.
# drop a column with string
df = df.drop('name', axis=1)
df.applymap(np.sqrt)

# first five rows
first5 = df.iloc[:5, :]
# return a subset
incomesubset = df[(df['applicant_income_000s'] > 0 ) & (df['applicant_income_000s'] < 1000)]
# shape of the dataset
df.shape
# Query the columns of a frame with a boolean expression.
df.query('a>b') # = df[df.a > df.b]























