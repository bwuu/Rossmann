import numpy as np
import pandas as pd
from rmspe import rmspe

data = pd.read_csv('../train_processed.csv',delimiter=',')
#select features
features = ['Store',
            'DayOfWeek',
            #'Sales',
            'Promo',
            'StateHoliday',
            'SchoolHoliday',
            'StoreType',
            'Assortment',
            'CompetitionDistance',
            'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear',
            'Promo2',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'year',
            'month',
            'day']

targets = data['Sales'].values
targets = np.reshape(targets, [targets.shape[0], 1])
data = data[features].values
data = np.concatenate((targets, data), axis=1)
np.random.shuffle(data)

from sklearn.ensemble import RandomForestRegressor
offset = int(data.shape[0] * 0.9)
train = data[:offset, :]
test = data[offset:, :]

params= {'n_estimators':30, 'n_jobs':2, 'verbose':2, 'max_depth':30, 'max_features':0.6,}
forest = RandomForestRegressor(**params)
forest.fit(train[:, 1::], np.log1p(train[:, 0]))

# WARNING: MAY DUMP A FEW GIGS OR MORE
# from sklearn.externals import joblib
# joblib.dump(forest, 'rf_001.model')
# print 'wrote forest model to rf_001.model'

out = np.expm1(forest.predict(test[:, 1::]))
target = test[:, 0]
err = rmspe(target, out)

print err
print 'done'

print forest.feature_importances_
