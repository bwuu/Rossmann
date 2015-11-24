__author__ = 'osboxes'

import numpy as np
import pandas as pd
from datetime import datetime

def process(data):
    # map categoricals to numbers
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    # date features
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # don't want to use customers from train data
    if 'Customers' in data:
        data = data.drop(['Customers'], axis=1)

    # stuff to be dropped regardless of train or test data
    data = data.drop(['PromoInterval', 'Open'], axis=1)

    #feature engineering
    data['CompetitionOpenDeltaMonths'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + (data.Month - data.CompetitionOpenSinceMonth)

    data['PromoOpenDeltaMonths'] = 12 * (data.Year - data.Promo2SinceYear) + (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpenDeltaMonths'] = data.PromoOpenDeltaMonths.apply(lambda x: x if x > 0 else 0)
    #data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    data = data.fillna(-999)
    return data

np.set_printoptions(suppress=True)

print("Load the training, test and store data using pandas")
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
train = pd.read_csv("../train.csv", parse_dates=[2], dtype=types)
test = pd.read_csv("../test.csv", parse_dates=[3], dtype=types)
store = pd.read_csv("../store.csv", dtype={'PromoInterval': np.dtype(str)})

# Parse PromoInterval
store['PromoInterval'] = store['PromoInterval'].replace(np.nan, '', regex=True)
store['PromoInterval'] = store['PromoInterval'].apply(lambda x: x.split(','))
store['JanPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Jan' in x else 0)
store['FebPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Feb' in x else 0)
store['MarPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Mar' in x else 0)
store['AprPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Apr' in x else 0)
store['MayPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'May' in x else 0)
store['JunPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Jun' in x else 0)
store['JulPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Jul' in x else 0)
store['AugPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Aug' in x else 0)
store['SeptPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Sept' in x else 0)
store['OctPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Oct' in x else 0)
store['NovPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Nov' in x else 0)
store['DecPromo'] = store['PromoInterval'].apply(lambda x: 1 if 'Dec' in x else 0)

train = pd.merge(train, store, on='Store')
train = train[train['Open'] != 0]
train = train[train['Sales'] > 0]
train = process(train)
print('training data processed')

# store 622
test['Open'].apply(lambda x: 1 if np.isnan(x) else x)
test = pd.merge(test, store, on='Store')
test = process(test)
print('testing data processed')
train.to_csv('../train_processed.csv', delimiter=',')
print('wrote processed training data to train_processed.csv')
print('columns: ' + str(train.columns.values.tolist()))
test.to_csv('../test_processed.csv', delimiter=',')
print('wrote processed testing data to test_processed.csv')
print('columns: ' + str(test.columns.values.tolist()))

#print train