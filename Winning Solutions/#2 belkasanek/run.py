#!/usr/bin/python3
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool

# Load files
DATA_PATH = 'data'
date_columns = ['CropTillageDate', 'RcNursEstDate', 'Harv_date', 
		'Threshing_date', 'SeedingSowingTransplanting'] 
train = pd.read_csv(os.path.join(DATA_PATH, 'Train.csv'), parse_dates=date_columns)
test = pd.read_csv(os.path.join(DATA_PATH, 'Test.csv'), parse_dates=date_columns)

# normalize continuous features by area
cols_to_norm = ['TransIrriCost', 'CropOrgFYM', 'Harv_hand_rent', 'Ganaura', 'BasalDAP', 			 'BasalUrea', '1tdUrea', '2tdUrea', 'CropCultLand', 'CultLand', 'Yield']
train[cols_to_norm] = train[cols_to_norm] / train['Acre'].values.reshape(-1, 1)
test[cols_to_norm[:-1]] = test[cols_to_norm[:-1]] / test['Acre'].values.reshape(-1, 1)

train['CropCultLand'] = train['CropCultLand'] / train['CultLand']
test['CropCultLand'] = test['CropCultLand'] / test['CultLand']

# fix mistypes
test.loc[test['Block'] == 'Lohra', 'Block'] = 'Jamui'
train.loc[(train['Block'] == 'Gurua') & (train['District'] == 'Jamui'), 'District'] = 'Gaya'

categorical_data = train.columns[train.dtypes == 'object'].values[1:]
categorical_data = list(categorical_data)

# fix target outliers
# too large
col = 'Yield'
limit = 4500
train.loc[train[col] >= limit, col] = train.loc[train[col] >= limit, col] / 10
# too small
limit = 130
train.loc[train[col] <= limit, col] = train.loc[train[col] <= limit, col] * 10

for col, limit in zip(['Harv_hand_rent', 'TransplantingIrrigationHours', 				'CultLand','SeedlingsPerPit'], 
                      [10000, 500, 1500, 100]):
    train.loc[train[col] >= limit, col] = train[col].median()
    test.loc[test[col] >= limit, col] = train[col].median()

# feature engineering
for i, col in enumerate(['RcNursEstDate', 'SeedingSowingTransplanting']):
    train[f'date_col{i}'] = train[col].dt.day_of_year // 7
    test[f'date_col{i}'] = test[col].dt.day_of_year // 7

# drop date columns
train.drop(date_columns, axis=1, inplace=True)
test.drop(date_columns, axis=1, inplace=True)

# impute NaN values for categorical features
train[categorical_data] = train[categorical_data].fillna('nan')
test[categorical_data] = test[categorical_data].fillna('nan')

def preprocess(x):
    x = x.split()
    x.sort()
    return ' '.join(x)

# these columns contains list of string
# sort words in them so 'a b' and 'b a' are treated same by model
for col in ['LandPreparationMethod', 'OrgFertilizers', 'TransDetFactor']:
    train[col] = train[col].apply(preprocess)
    test[col] = test[col].apply(preprocess)
    
X = train.drop(['ID', 'Yield'], axis=1)
X_test = test.drop(['ID'], axis=1)
y = train['Yield']

cat_idx = []
columns = list(X.columns)
for cat_col in categorical_data:
    if cat_col in columns:
        cat_idx.append(columns.index(cat_col))
    
test_pred = np.zeros(len(test))

for num_model in range(3):
    kf = KFold(n_splits=5, shuffle=True, random_state=hash(num_model*3))
    kf = kf.split(X)
    
    for i, (train_idx, valid_idx) in enumerate(kf):
        train_df = X.iloc[train_idx]
        valid_df = X.iloc[valid_idx]
        
        y_train = y.iloc[train_idx]
        y_val = y.iloc[valid_idx]
        
        train_pool = Pool(train_df, y_train, cat_features=cat_idx)
        val_pool = Pool(valid_df, y_val, cat_features=cat_idx) 
        test_pool = Pool(X_test, cat_features=cat_idx) 
        
        model = CatBoostRegressor(iterations=400, 
                                  depth=10, 
                                  learning_rate=0.15, 
                                  min_data_in_leaf=16,
                                  border_count=128,
                                  early_stopping_rounds=30,
                                  loss_function='RMSE',
                                  verbose=0)
        
        model.fit(X=train_pool, eval_set=val_pool)
        test_pred += (model.predict(test_pool) * X_test['Acre']) / 15   

    
sub = pd.DataFrame({'ID': test['ID'], 'Yield': test_pred})
sub.to_csv('solution.csv', index=False)
