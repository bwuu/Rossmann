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
            'year',
            'month',
            'day',
            'JanPromo',
            'FebPromo',
            'MarPromo',
            'AprPromo',
            'MayPromo',
            'JunPromo',
            'JulPromo',
            'SeptPromo',
            'OctPromo',
            'NovPromo',
            'DecPromo']

# feature engineering
compDataAvailable = (data.CompetitionOpenSinceYear != 0) & (data.CompetitionOpenSinceMonth != 0)
getCompetitionDeltaDays = lambda x: (datetime(int(x.CompetitionOpenSinceYear), int(x.CompetitionOpenSinceMonth), 1) - x.Date).days
competitionOpenDeltaDays = data[compDataAvailable].apply(getCompetitionDeltaDays, axis=1)
data['CompetitionOpenDeltaDays'] = competitionOpenDeltaDays
data = data.fillna(-9999)

targets = data['Sales'].values
targets = np.reshape(targets, [targets.shape[0], 1])
data = data[features].values
data = np.concatenate((targets, data), axis=1)

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.09,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          #"silent": 1
          "seed": 1301
          }
num_boost_round = 1000
print("Train an XGBoost model")
X_train, X_valid = train_test_split(data, test_size=0.012, random_state=10)

y_train = np.log1p(X_train[:, 0])
y_valid = np.log1p(X_valid[:, 0])
dtrain = xgb.DMatrix(X_train[:, 1:], y_train)
dvalid = xgb.DMatrix(X_valid[:, 1:], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=10, feval=rmspe_xg, verbose_eval=True)

gbm.save_model('../saved_models/xgb_001.model')
print 'model saved in xgb_001.model'

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[:, 1:]))
error = rmspe(X_valid[:,0], np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))
#

# print("Make predictions on the test set")
# dtest = xgb.DMatrix(test)
# test_probs = gbm.predict(dtest)
# Make Submission
# result = pd.DataFrame({"Id": testIds, 'Sales': np.expm1(test_probs)})
# result.to_csv("/home/osboxes/Desktop/xgboost_10_submission.csv", index=False)

# XGB feature importances
# Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

# importance = gbm.get_fscore(fmap='xgb.fmap')
# importance = sorted(importance.items(), key=operator.itemgetter(1))

# df = pd.DataFrame(importance, columns=['feature', 'fscore'])
# df['fscore'] = df['fscore'] / df['fscore'].sum()

# featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# fig_featp = featp.get_figure()
# fig_featp.savefig('/home/osboxes/Desktopfeature_importance_xgb.png', bbox_inches='tight', pad_inches=1)
