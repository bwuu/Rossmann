from sklearn.cross_validation import train_test_split

__author__ = 'osboxes'

import numpy as np
import pandas as pd
import xgboost as xgb
from rmspe import rmspe
from rmspe import rmspe_xg
from datetime import datetime

data = pd.read_csv('../train_processed.csv',delimiter=',', parse_dates=[3], index_col=0)
#select features
features = ['Store',
            'DayOfWeek',
            #'Date'
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
            'Year',
            'Month',
            'Day',
            #'JanPromo',
            #'FebPromo',
            #'MarPromo',
            #'AprPromo',
            #'MayPromo',
            #'JunPromo',
            #'JulPromo',
            #'SeptPromo',
            #'OctPromo',
            #'NovPromo',
            #'DecPromo',
            'CompetitionOpenDeltaMonths',
            'PromoOpenDeltaMonths'
            ]

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.09,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          #"silent": 1
          "seed": 1301
          }
num_boost_round = 1700
print("Train an XGBoost model")

# sales in first column for easy syntax later
holdout = (data.Year==2014) & ((data.Month==8))
X_train = data[['Sales'] + features][~holdout].values
X_valid = data[['Sales'] + features][holdout].values

#data = data[['Sales'] + features].values
#X_train, X_valid = train_test_split(data, test_size=0.012, random_state=10)

y_train = np.log1p(X_train[:, 0])
y_valid = np.log1p(X_valid[:, 0])
dtrain = xgb.DMatrix(X_train[:, 1:], y_train)
dvalid = xgb.DMatrix(X_valid[:, 1:], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=120, feval=rmspe_xg, verbose_eval=True)

gbm.save_model('../saved_models/xgb_001.model')
print 'model saved in xgb_001.model'

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[:, 1:]))
error = rmspe(X_valid[:,0], np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))