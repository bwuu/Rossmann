__author__ = 'osboxes'

import numpy as np
import pandas as pd

def process(data):
    # map categoricals to numbers
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    # date features
    data['year'] = data.Date.dt.year
    data['month'] = data.Date.dt.month
    data['day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek

    # don't want to use customers from train data
    if 'Customers' in data:
        data = data.drop(['Customers'], axis=1)

    # stuff to be dropped regardless of train or test data
    data = data.drop(['Date', 'PromoInterval', 'Open'], axis=1)

    data = data.fillna(0)

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
store = pd.read_csv("../store.csv")

train = pd.merge(train, store, on='Store')
train = train[train['Open'] != 0]
train = train[train['Sales'] > 0]
train = process(train)
print('training data processed')

test = pd.merge(test, store, on='Store')
test = test[test['Open'] != 0]
test = process(test)
print('testing data processed')
train.to_csv('../train_processed.csv', delimiter=',')
print('wrote processed training data to train_processed.csv')
print('columns: ' + str(train.columns.values.tolist()))
test.to_csv('../test_processed.csv', delimiter=',')
print('wrote processed testing data to test_processed.csv')
print('columns: ' + str(test.columns.values.tolist()))

#print train