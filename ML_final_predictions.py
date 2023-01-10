import pickle
import pandas as pd
import math
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import jupytext


# Load data

df_touse, df_predict = pickle.load(open('datasets.sav','rb'))
df = df_touse


predictors = df.columns.tolist()
predictors.remove("target")
X = df[predictors]
y = df["target"]

# Splitting dataset and storing descriptive statsitics and indices of splits to process whole dataframe to prevent data leakage

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2)

test_index_values = X_test.index.values
train_index_values = Y_train.index.values


# Select only the numerical columns
numeric_cols = X_train.select_dtypes(include=['int', 'float'])

# Compute the mean of each numerical column
X_train_mean = numeric_cols.mean()
X_test_mean = numeric_cols.mean()


# Compute the mean of each numerical column
Y_train_mean = Y_train.mean()


# Compute the mean of each numerical column
Y_test_mean = Y_test.mean()



# Transform data
## numerical NaNs replaced by Mean of respective split

df = df.drop(columns=['insee'])

df.loc[train_index_values, ['PERSON_ID', 'AGE_2018', 'PAY', 'Working_hours']] = df.loc[train_index_values, ['PERSON_ID', 'AGE_2018', 'PAY', 'Working_hours']].fillna(X_train_mean)

df.loc[test_index_values, ['PERSON_ID', 'AGE_2018', 'PAY', 'Working_hours']] = df.loc[test_index_values, ['PERSON_ID', 'AGE_2018', 'PAY', 'Working_hours']].fillna(X_test_mean)

df.loc[train_index_values, ['target']] = df.loc[train_index_values, ['target']].fillna(Y_train_mean)

df.loc[test_index_values, ['target']] = df.loc[test_index_values, ['target']].fillna(Y_test_mean)



## categorical NaNs replaced by random category of variable while respecting proportions of respective split


# Select only the categorical columns
cat_cols = df.select_dtypes(include=['object'])

## For the training set

# Iterate over the categorical columns
for col_name in cat_cols.columns:
    # Select the column
    col = X_train[col_name]
    
    # Compute the proportions of the categories
    counts = col.value_counts(normalize=True)
    
    # Replace the NaN values with random categories
    for i in col.index:
        if pd.isnull(col[i]):
            col[i] = random.choices(counts.index, counts.values)[0]
            
df.update(col)


## For the test set

for col_name in cat_cols.columns:
    # Select the column
    col = X_test[col_name]
    
    # Compute the proportions of the categories
    counts = col.value_counts(normalize=True)
    
    # Replace the NaN values with random categories
    for i in col.index:
        if pd.isnull(col[i]):
            col[i] = random.choices(counts.index, counts.values)[0]
            
df.update(col)



## One Hot Encode categorical variables

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Select the string columns
string_columns = df.select_dtypes(['object']).columns

# Create a OneHotEncoder object
encoder = OneHotEncoder()

# Fit the encoder to the string columns
encoder.fit(df[string_columns])

# Encode the string columns
encoded_data = encoder.transform(df[string_columns])

# Create a DataFrame with the encoded columns and the original index
encoded_df = pd.DataFrame(encoded_data.toarray(), index=df.index, columns=encoder.get_feature_names(string_columns))

# Concatenate the encoded columns with the rest of the data
df = pd.concat([encoded_df, df.drop(string_columns, axis=1)], axis=1)


## Reassign processed splits 

X_train = df.loc[train_index_values]
X_train = X_train.drop('target', axis=1)
X_test =  df.loc[test_index_values]
X_test = X_test.drop('target', axis=1)
Y_train = df.loc[train_index_values]['target']
Y_test = df.loc[test_index_values]['target']



# Models

## Decision Tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold

dt_params = {'min_samples_split': [2, 5] + list(range(10, 100, 5))}

#,
#              'max_features': ['auto', 'sqrt', 'log2', None],
#              'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3]}


#dt_params = {'min_samples_split': [2, 5] + list(range(10, 250,5))} 
dt = DecisionTreeRegressor(random_state=0)
cv_folds = KFold(5, shuffle=True, random_state=0)
dt_cv = GridSearchCV(dt, dt_params, cv=cv_folds, n_jobs=-1) 
dt_cv.fit(X_train, Y_train) 
print(dt_cv.best_score_)
print(dt_cv.best_params_)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Make predictions on the test data
predictions_dt = dt_cv.predict(X_test)

# Compute the mean absolute error
mae = mean_absolute_error(Y_test, predictions_dt)

# Compute the mean squared error
mse = mean_squared_error(Y_test, predicpredictions_dttions)

# Compute the root mean squared error
rmse = math.sqrt(mse)

# Compute the R-squared score
r2 = r2_score(Y_test, predictions_dt)

# Compute the adjusted R-squared score
n = len(Y_test)
p = X_test.shape[1]  # number of features
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)



print("Mean absolute error: {:.2f}".format(mae))
print("Mean squared error: {:.2f}".format(mse))
print("Root mean squared error: {:.2f}".format(rmse))
print("R-squared score: {:.2f}".format(r2))
print("Adjusted R-squared score: {:.2f}".format(adj_r2))




## Gradient Boosting


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV

# Define the grid of hyperparameters to search
param_grid = {'learning_rate': [0.2,0.1, 0.05, 0.01, 0.001]}


# Create the gradient boosting model
gb = GradientBoostingRegressor(random_state=0)

# Create the K-fold cross-validation object
cv_folds = KFold(5, shuffle=True, random_state=0)

# Create the grid search object
gb_cv = GridSearchCV(gb, param_grid, cv=cv_folds, n_jobs=-1)

# Fit the grid search object to the training data
gb_cv.fit(X_train, Y_train)

print(gb_cv.best_score_)
print(gb_cv.best_params_)


# Make predictions on the test data
predictions_gb = gb_cv.predict(X_test)

# Compute the mean absolute error
mae = mean_absolute_error(Y_test, predictions_gb)

# Compute the mean squared error
mse = mean_squared_error(Y_test, predictions_gb)

# Compute the root mean squared error
rmse = math.sqrt(mse)

# Compute the R-squared score
r2 = r2_score(Y_test, predictions_gb)

# Compute the adjusted R-squared score
n = len(Y_test)
p = X_test.shape[1]  # number of features
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)



print("Mean absolute error: {:.2f}".format(mae))
print("Mean squared error: {:.2f}".format(mse))
print("Root mean squared error: {:.2f}".format(rmse))
print("R-squared score: {:.2f}".format(r2))
print("Adjusted R-squared score: {:.2f}".format(adj_r2))


## Ridge

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV

# Define the grid of hyperparameters to search
param_grid = {'alpha': [0.1, 1.0, 10.0, 15.0, 100.0, 1000.0]}
              #,'max_iter': [100, 1000, 10000, 100000]}

# Create the ridge regression model
ridge = Ridge(random_state=0)

# Create the K-fold cross-validation object
cv_folds = KFold(5, shuffle=True, random_state=0)

# Create the grid search object
ridge_cv = GridSearchCV(ridge, param_grid, cv=cv_folds, n_jobs=-1)

# Fit the grid search object to the training data
ridge_cv.fit(X_train, Y_train)
print(ridge_cv.best_score_)
print(ridge_cv.best_params_)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Make predictions on the test data
predictions_ridge = ridge_cv.predict(X_test)

# Compute the mean absolute error
mae = mean_absolute_error(Y_test, predictions_ridge)

# Compute the mean squared error
mse = mean_squared_error(Y_test, predictions_ridge)

# Compute the root mean squared error
rmse = math.sqrt(mse)

# Compute the R-squared score
r2 = r2_score(Y_test, predictions_ridge)

# Compute the adjusted R-squared score
n = len(Y_test)
p = X_test.shape[1]  # number of features
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)



print("Mean absolute error: {:.2f}".format(mae))
print("Mean squared error: {:.2f}".format(mse))
print("Root mean squared error: {:.2f}".format(rmse))
print("R-squared score: {:.2f}".format(r2))
print("Adjusted R-squared score: {:.2f}".format(adj_r2))




## Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV

# Define the grid of hyperparameters to search
param_grid = {'n_estimators': [5,7,10]}
              #'max_depth': [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              #'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              #'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

# Create the random forest model
rf = RandomForestRegressor(random_state=0)

# Create the K-fold cross-validation object
cv_folds = KFold(5, shuffle=True, random_state=0)

# Create the grid search object
rf_cv = GridSearchCV(rf, param_grid, cv=cv_folds, n_jobs=-1)

# Fit the grid search object to the training data
rf_cv.fit(X_train, Y_train)
print(rf_cv.best_score_)
print(rf_cv.best_params_)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Make predictions on the test data
predictions_rf = rf_cv.predict(X_test)

# Compute the mean absolute error
mae = mean_absolute_error(Y_test, predictions_rf)

# Compute the mean squared error
mse = mean_squared_error(Y_test, predictions_rf)

# Compute the root mean squared error
rmse = math.sqrt(mse)

# Compute the R-squared score
r2 = r2_score(Y_test, predictions_rf)

# Compute the adjusted R-squared score
n = len(Y_test)
p = X_test.shape[1]  # number of features
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)



print("Mean absolute error: {:.2f}".format(mae))
print("Mean squared error: {:.2f}".format(mse))
print("Root mean squared error: {:.2f}".format(rmse))
print("R-squared score: {:.2f}".format(r2))
print("Adjusted R-squared score: {:.2f}".format(adj_r2))


## Lasso

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, GridSearchCV

# Define the grid of hyperparameters to search
param_grid = {'alpha': [0.005, 0.001, 0.00005]}
              #,'max_iter': [100, 1000, 10000, 100000]}

# Create the Lasso regression model
lasso = Lasso(random_state=0)

# Create the K-fold cross-validation object
cv_folds = KFold(5, shuffle=True, random_state=0)

# Create the grid search object
lasso_cv = GridSearchCV(lasso, param_grid, cv=cv_folds, n_jobs=-1)

# Fit the grid search object to the training data
lasso_cv.fit(X_train, Y_train)
print(lasso_cv.best_score_)
print(lasso_cv.best_params_)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Make predictions on the test data
predictions_lasso = lasso_cv.predict(X_test)

# Compute the mean absolute error
mae = mean_absolute_error(Y_test, predictions_lasso)

# Compute the mean squared error
mse = mean_squared_error(Y_test, predictions_lasso)

# Compute the root mean squared error
rmse = math.sqrt(mse)

# Compute the R-squared score
r2 = r2_score(Y_test, predictions_lasso)

# Compute the adjusted R-squared score
n = len(Y_test)
p = X_test.shape[1]  # number of features
adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)



print("Mean absolute error: {:.2f}".format(mae))
print("Mean squared error: {:.2f}".format(mse))
print("Root mean squared error: {:.2f}".format(rmse))
print("R-squared score: {:.2f}".format(r2))
print("Adjusted R-squared score: {:.2f}".format(adj_r2))









##########################

#Prediction of df_predict dataset

# Transform data

df_predict = df_predict.drop(columns=['insee'])

### numerical NaNs replaced by Mean


## for the df_touse or train set
# Select only the numerical columns
numeric_cols = df_touse.select_dtypes(include=['int', 'float'])

# Compute the mean of each numerical column
means = numeric_cols.mean()

# Replace NaN values with the median of each column
df_touse[numeric_cols.columns] = numeric_cols.fillna(means)


## for the df_predict set
# Select only the numerical columns
numeric_cols = df_predict.select_dtypes(include=['int', 'float'])

# Compute the mean of each numerical column
means = numeric_cols.mean()

# Replace NaN values with the median of each column
df_predict[numeric_cols.columns] = numeric_cols.fillna(means)



## transform both datasetss' categorical NaNs replaced by random category of variable while respecting proportions

cat_cols = df_predict.select_dtypes(include=['object'])

### for the df_touse or trainset
# Iterate over the categorical columns
for col_name in cat_cols.columns:
    # Select the column
    col = df_touse[col_name]
    
    # Compute the proportions of the categories
    counts = col.value_counts(normalize=True)
    
    # Replace the NaN values with random categories
    for i in col.index:
        if pd.isnull(col[i]):
            col[i] = random.choices(counts.index, counts.values)[0]
            
df_touse.update(col)

### for the df_predict or testset
# Iterate over the categorical columns
for col_name in cat_cols.columns:
    # Select the column
    col = df_predict[col_name]
    
    # Compute the proportions of the categories
    counts = col.value_counts(normalize=True)
    
    # Replace the NaN values with random categories
    for i in col.index:
        if pd.isnull(col[i]):
            col[i] = random.choices(counts.index, counts.values)[0]
            
df_predict.update(col)







## One Hot Encode categorical variables
### for the df_touse or trainset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


### Select the string columns
string_columns = df_touse.select_dtypes(['object']).columns

### Create a OneHotEncoder object
encoder = OneHotEncoder()

### Fit the encoder to the string columns
encoder.fit(df_touse[string_columns])

### Encode the string columns
encoded_data = encoder.transform(df_touse[string_columns])

### Create a DataFrame with the encoded columns and the original index
encoded_df = pd.DataFrame(encoded_data.toarray(), index=df_touse.index, columns=encoder.get_feature_names(string_columns))

### Concatenate the encoded columns with the rest of the data
df_touse = pd.concat([encoded_df, df_touse.drop(string_columns, axis=1)], axis=1)


### for the df_predict or testset

### Select the string columns
string_columns = df_predict.select_dtypes(['object']).columns

### Create a OneHotEncoder object
encoder = OneHotEncoder()

### Fit the encoder to the string columns
encoder.fit(df_predict[string_columns])

### Encode the string columns
encoded_data = encoder.transform(df_predict[string_columns])

### Create a DataFrame with the encoded columns and the original index
encoded_df = pd.DataFrame(encoded_data.toarray(), index=df_predict.index, columns=encoder.get_feature_names(string_columns))

### Concatenate the encoded columns with the rest of the data
df_predict = pd.concat([encoded_df, df_predict.drop(string_columns, axis=1)], axis=1)






# Train Gradiant Boosting model



### Make predictions on the dt_touse or test data
Y = df_touse['target']
X = df_touse.drop('target', axis=1)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV

# Define the grid of hyperparameters to search
param_grid = {'learning_rate': [0.2,0.1, 0.05, 0.01, 0.001]}


# Create the gradient boosting model
gb = GradientBoostingRegressor(random_state=0)

# Create the K-fold cross-validation object
cv_folds = KFold(5, shuffle=True, random_state=0)

# Create the grid search object
gb_cv = GridSearchCV(gb, param_grid, cv=cv_folds, n_jobs=-1)

# Fit the grid search object to the training data
gb_cv.fit(X, Y)

predictions_gb = gb_cv.predict(df_predict)




predictions = df_predict['PERSON_ID']
predictions = predictions.assign(target=predictions_gb)

predictions.to_csv('predictions.csv', index=False)
