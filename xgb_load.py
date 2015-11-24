import operator
import matplotlib.pyplot as plt

__author__ = 'osboxes'

import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime

gbm = xgb.Booster()
gbm.load_model('saved_models/xgb_001.model')

test = pd.read_csv('test_processed.csv',delimiter=',', parse_dates=[4], index_col=0)
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

data = test[features].values

print("Make predictions on the test set")
dtest = xgb.DMatrix(data)
test_probs = gbm.predict(dtest)
# Make Submission
result = pd.DataFrame({"Id": test['Id'], 'Sales': np.expm1(test_probs)})
result.to_csv("xgboost_submission.csv", index=False)

# XGB feature importances
# Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code

importance = gbm.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)