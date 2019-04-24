import numpy as np
import pandas as pd
import random
import csv
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import xgboost as xgb

df = pd.read_csv('1904192125-pred_table.csv')
blocks = df['block'].unique()
print('number of blocks:', len(blocks))

quantiles = df['Gas Price (Gwei)'].quantile([0.25, 0.5, 0.75])
# print('quantiles:\n', quantiles)

df['gasPriceCat1'] = (df['Gas Price (Gwei)'] <= quantiles[0.25]).astype(int)
# print(df['gasPriceCat1'])

df['gasPriceCat2'] = ((df['Gas Price (Gwei)'] > quantiles[0.25]) & (df['Gas Price (Gwei)'] <= quantiles[0.5])).astype(
    int)

df['gasPriceCat3'] = ((df['Gas Price (Gwei)'] > quantiles[0.5]) & (df['Gas Price (Gwei)'] <= quantiles[0.75])).astype(
    int)

df['gasPriceCat4'] = (df['Gas Price (Gwei)'] > quantiles[0.75]).astype(int)

y = df['Mean Time to Confirm (Minutes)'].values
for i in range(len(y)):
    if y[i] == '> 2 hours':
        y[i] = 120
    else:
        y[i] = float(y[i])

gs_pred = df['95% Confidence Confirm Time (Minutes)'].values
for i in range(len(gs_pred)):
    if gs_pred[i] == '> 2 hours':
        gs_pred[i] = 120
    else:
        gs_pred[i] = float(gs_pred[i])

x_0 = df['Gas Price (Gwei)'].values
x_1 = df['gasPriceCat1'].values
x_2 = df['gasPriceCat2'].values
x_3 = df['gasPriceCat3'].values
x_4 = df['gasPriceCat4'].values
x_5 = df['% of Last 200 Blocks Accepting'].values
x_6 = df['#Tx at/above in txpool'].values
x_7 = df['block'].values
X = [x_1, x_2, x_3, x_4, x_5, x_6, x_7]
# X = [x_0, x_5, x_6, x_7]
X = np.array(X).T

print('X.shape:', X.shape)
print('y.shape:', y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def MultipleLinearRegression(X, y, linear_model):
    lm = linear_model
    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X)

    new_X = np.append(np.ones((len(X), 1)), X, axis=1)
    MSE = (sum((y - predictions) ** 2)) / (len(new_X) - len(new_X[0]))

    var_b = MSE * (np.linalg.inv(np.dot(new_X.T, new_X)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(new_X) - 1))) for i in ts_b]

    table = pd.DataFrame()
    table["Coefficients"], table["Standard Errors"], table["t values"], table["p values"] = [params, sd_b, ts_b,
                                                                                             p_values]
    print(table)


linear_model = LinearRegression().fit(x_train, y_train)
MultipleLinearRegression(x_train, y_train, linear_model)

### train
y_hat = linear_model.predict(x_train)
r_squared = metrics.r2_score(y_train, y_hat)
print('Training R-Squared:', r_squared)

mse_train = metrics.mean_squared_error(y_train, y_hat)
print('Training MSE:', mse_train)

### test
y_pred = linear_model.predict(x_test)
mse_test = metrics.mean_squared_error(y_test, y_pred)
print('Testing MSE:', mse_test)

### XGBoost
reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                       colsample_bytree=1, max_depth=7)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
mse_test = metrics.mean_squared_error(y_test, y_pred)
print('XGBoost Testing MSE:', mse_test)

### gas station
mse_gs = metrics.mean_squared_error(y, gs_pred)
print('Gas station MSE:', mse_gs)
