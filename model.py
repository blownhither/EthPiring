import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from scipy import stats
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR

df = pd.read_csv('1904230522-pred_table.csv')
# df = pd.read_csv('1904192125-pred_table.csv')

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

blocks = set(x_7)

print('X.shape:', X.shape)
print('y.shape:', y.shape)

###### static model
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
# def MultipleLinearRegression(X, y, linear_model):
#     lm = linear_model
#     params = np.append(lm.intercept_, lm.coef_)
#     predictions = lm.predict(X)
#
#     new_X = np.append(np.ones((len(X), 1)), X, axis=1)
#     MSE = (sum((y - predictions) ** 2)) / (len(new_X) - len(new_X[0]))
#
#     var_b = MSE * (np.linalg.inv(np.dot(new_X.T, new_X)).diagonal())
#     sd_b = np.sqrt(var_b)
#     ts_b = params / sd_b
#
#     p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(new_X) - 1))) for i in ts_b]
#
#     table = pd.DataFrame()
#     table["Coefficients"], table["Standard Errors"], table["t values"], table["p values"] = [params, sd_b, ts_b,
#                                                                                              p_values]
#     print(table)
#
#
# linear_model = LinearRegression().fit(x_train, y_train)
# MultipleLinearRegression(x_train, y_train, linear_model)
#
# ### train
# y_hat = linear_model.predict(x_train)
# r_squared = metrics.r2_score(y_train, y_hat)
# print('Training R-Squared:', r_squared)
#
# mse_train = metrics.mean_squared_error(y_train, y_hat)
# print('Training MSE:', mse_train)
#
# ### test
# y_pred = linear_model.predict(x_test)
# mse_test = metrics.mean_squared_error(y_test, y_pred)
# print('Testing MSE:', mse_test)
#
# ### XGBoost
# reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
#                        colsample_bytree=1, max_depth=7)
# reg.fit(x_train, y_train)
# y_pred = reg.predict(x_test)
# mse_test = metrics.mean_squared_error(y_test, y_pred)
# print('XGBoost Testing MSE:', mse_test)
#
# ### gas station
# mse_gs = metrics.mean_squared_error(y, gs_pred)
# print('Gas station MSE:', mse_gs)

####### sliding window model

# number of BLOCKs instead of samples
train_size = 2000
window_size = train_size * 0.5


def find_train_blocks(cur, blocks, train_size, X, y, gs_pred):
    """
    :param cur: current block id
    :param blocks: list of all block ids
    :param train_size: size of training data (backwards)
    :param X: features
    :param y: labels
    :param gs_pred: predicted confirmation time by gas station
    :return:
    """
    res = []
    block_dict = {}
    count = 0
    for j in reversed(range(cur)):
        if count == train_size:
            break
        block_id = X[j][-1]
        if block_id in blocks:
            block_of_samples = find_block_by_id(block_id, X, y, gs_pred)
            block_dict[j] = block_of_samples
            for sample in block_of_samples:
                res.append(sample)
        count += 1
    res = np.array(res)
    return block_dict, res


def find_test_blocks(cur, blocks, window_size, X, y, gs_pred):
    res = []
    block_dict = {}
    count = 0
    for j in range(cur + 1, len(blocks)):
        if count == window_size:
            break
        block_id = X[j][-1]
        if block_id in blocks:
            block_of_samples = find_block_by_id(block_id, X, y, gs_pred)
            block_dict[j] = block_of_samples
            for sample in block_of_samples:
                res.append(sample)
        count += 1
    res = np.array(res)
    return block_dict, res


def find_block_by_id(id, X, y, gs_pred):
    """
    :param id: block id to be found
    :param X: features
    :param y: labels
    :param gs_pred: predicted confirmation time by gas station
    :return:
    """
    res = []
    for j in range(len(X)):
        if X[j][-1] == id:
            res.append(np.concatenate([X[j], [y[j]], [gs_pred[j]]]))
    return np.array(res)


def train_predict(start, blocks, train_size, window_size, X, y, gs_pred, model):
    """
    :param start: starting index of the first sliding window
    :param blocks: list of all block ids
    :param train_size: size of training data (backwards)
    :param window_size: size of testing data
    :param X: features
    :param y: labels
    :param gs_pred: predicted confirmation time by gas station
    :param model: ML model name
    :return:
    """
    mse_hist = []
    count = 0
    for j in range(start, len(blocks)):

        if j + window_size > len(blocks) or count >= 11:
            break
        _, train = find_train_blocks(j, blocks, train_size, X, y, gs_pred)
        _, test = find_test_blocks(j, blocks, window_size, X, y, gs_pred)

        x_train = train[:, :-2]
        y_train = train[:, -2]
        x_test = test[:, :-2]
        y_test = test[:, -2]
        gs_test = test[:, -1]
        if model == 'xgb':
            reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                                   colsample_bytree=1, max_depth=7)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            mse_test = metrics.mean_squared_error(y_test, y_pred)
            mse_hist.append(mse_test)
            print(mse_hist)
        elif model == 'ada':
            reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=0)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            mse_test = metrics.mean_squared_error(y_test, y_pred)
            mse_hist.append(mse_test)
            print(mse_hist)
        elif model == 'linear':
            linear_model = LinearRegression().fit(x_train, y_train)
            y_pred = linear_model.predict(x_test)
            mse_test = metrics.mean_squared_error(y_test, y_pred)
            mse_hist.append(mse_test)
            print(mse_hist)
        elif model == 'gs':
            mse_gs = metrics.mean_squared_error(y_test, gs_test)
            mse_hist.append(mse_gs)
            print(mse_hist)
        elif model == 'svm':
            reg = SVR(C=1.0, epsilon=0.2)
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_test)
            mse_test = metrics.mean_squared_error(y_test, y_pred)
            mse_hist.append(mse_test)
            print(mse_hist)
        # elif model == 'svm_linear':
        #     reg = LinearSVR(random_state=0, tol=1e-5)
        #     reg.fit(x_train, y_train)
        #     y_pred = reg.predict(x_test)
        #     mse_test = metrics.mean_squared_error(y_test, y_pred)
        #     mse_hist.append(mse_test)
        #     print(mse_hist)
        # elif model == 'lasso':
        #     reg = Lasso(alpha=0.1)
        #     reg.fit(x_train, y_train)
        #     y_pred = reg.predict(x_test)
        #     mse_test = metrics.mean_squared_error(y_test, y_pred)
        #     mse_hist.append(mse_test)
        #     print(mse_hist)
        # elif model == 'ridge':
        #     reg = Ridge(alpha=0.1)
        #     reg.fit(x_train, y_train)
        #     y_pred = reg.predict(x_test)
        #     mse_test = metrics.mean_squared_error(y_test, y_pred)
        #     mse_hist.append(mse_test)
        #     print(mse_hist)

        j += window_size
        count += 1
    return mse_hist


mse_hist = train_predict(5000, blocks, train_size, window_size, X, y, gs_pred, 'svm')
print('mse_hist:', mse_hist)
