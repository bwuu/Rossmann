from sklearn.cross_validation import train_test_split

__author__ = 'osboxes'

import xgboost as xgb
import numpy as np
from rmspe import rmspe
from rmspe import rmspe_xg

gbm = xgb.Booster()
gbm.load_model('xgb_001.model')

train = np.loadtxt('train_processed.csv',delimiter=',')

X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)

y_train = np.log1p(X_train[:, 0])
y_valid = np.log1p(X_valid[:, 0])
dtrain = xgb.DMatrix(X_train[:, 1:], y_train)
dvalid = xgb.DMatrix(X_valid[:, 1:], y_valid)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[:, 1:]))
error = rmspe(X_valid[:,0], np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))