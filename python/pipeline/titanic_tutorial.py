# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:56:33 2016

http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html


"""

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

data = pd.read_csv('/home/hongbin/dataset/kaggle/titanic/train.csv')
data.head()
# fill in na value with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# visualize the suvival based gender
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))

# visualize the suvival based age
figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

# Feature engineering
# combine train and test
def get_combined_data():
    # reading train data
    train = pd.read_csv('../../data/train.csv')
    
    # reading test data
    test = pd.read_csv('../../data/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    
    return combined

combined = get_combined_data()
combined.shape

# extract title
def get_titles():
    '''
    map to run func to data series
    '''

    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)

# fill the missing age by median value of seperated groups
grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.median()

def fillAges(row):
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return 30
        elif row['Title'] == 'Mrs':
            return 45
        elif row['Title'] == 'Officer':
            return 49
        elif row['Title'] == 'Royalty':
            return 39
# have more branches above but ommit
combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

# process the names- drop name column and encode the titles
def process_names():
    
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    # one-hot encoding of categerical data, add at the end of each row. 
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)

# Processing Fare
def process_fares():
    
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)

# Processing Embarked
def process_embarked():
    
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)

# Processing Cabin
def process_cabin():
    
    global combined
    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    
    combined = pd.concat([combined,cabin_dummies],axis=1)
    
    combined.drop('Cabin',axis=1,inplace=True)
    
combined.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 1309 entries, 0 to 1308
#Data columns (total 26 columns):
#PassengerId      1309 non-null int64
#Pclass           1309 non-null int64
#Sex              1309 non-null object
#Age              1309 non-null float64
#SibSp            1309 non-null int64
#Parch            1309 non-null int64
#Ticket           1309 non-null object
#Fare             1309 non-null float64
#Title_Master     1309 non-null float64
#Title_Miss       1309 non-null float64
#Title_Mr         1309 non-null float64
#Title_Mrs        1309 non-null float64
#Title_Officer    1309 non-null float64
#Title_Royalty    1309 non-null float64
#Embarked_C       1309 non-null float64
#Embarked_Q       1309 non-null float64
#Embarked_S       1309 non-null float64
#Cabin_A          1309 non-null float64
#Cabin_B          1309 non-null float64
#Cabin_C          1309 non-null float64
#Cabin_D          1309 non-null float64
#Cabin_E          1309 non-null float64
#Cabin_F          1309 non-null float64
#Cabin_G          1309 non-null float64
#Cabin_T          1309 non-null float64
#Cabin_U          1309 non-null float64
#dtypes: float64(20), int64(4), object(2)
#memory usage: 266.0+ KB

# Processing Sex
def process_sex():
    
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})

# Processing Pclass
def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    
    # removing "Pclass"
    
    combined.drop('Pclass',axis=1,inplace=True)


# Processing Ticket
def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

# Processing Family
# everything convert to 1 or 0 indicator
def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)

# normalize
def scale_all_features():
    
    global combined
    
    features = list(combined.columns) # get column name
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print('Features scaled successfully !')
    
# III - Modeling
#1. Break the combined dataset in train set and test set.
#2. Use the train set to build a predictive model.
#3. Evaluate the model using the train set.
#4. Test the model using the test set and generate and output file for the submission.
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)

# get train, test and target
def recover_train_test_target():
    global combined
    
    train0 = pd.read_csv('../../data/train.csv')
    
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    
    return train,test,targets

train,test,targets = recover_train_test_target()

# Feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)


# create new df to hold and organize data
features = pd.DataFrame()
features['feature'] = train.columns # set  value of a column using a list
features['importance'] = clf.feature_importances_

features.sort(['importance'],ascending=False)

# select feature
model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
train_new.shape

test_new = model.transform(test)
test_new.shape

# Hyperparameters tuning
forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# generate output
output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('../../data/output.csv',index=False)


# pandas make preprocessing and postprecessing easy. 

















