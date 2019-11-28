import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import numpy as np


def sktime_to_numpy(train_file, test_file):
    X_train_ts, y_train = load_from_tsfile_to_dataframe(train_file)
    X_test_ts, y_test = load_from_tsfile_to_dataframe(test_file)
    train_max_len = max([len(X_train_ts.iloc[i]['dim_0']) for i in range(len(X_train_ts))])
    test_max_len = max([len(X_test_ts.iloc[i]['dim_0']) for i in range(len(X_test_ts))])
    max_len = max(train_max_len, test_max_len)

    X_train = np.zeros((len(X_train_ts), len(X_train_ts.columns), max_len))
    X_test = np.zeros((len(X_test_ts), len(X_test_ts.columns), max_len))
    for i in range(len(X_train_ts)):
        for col_idx, col in enumerate(X_train_ts.columns):
            X_train[i, col_idx] = np.pad(X_train_ts.iloc[i][col].values, pad_width=(0,max_len-len(X_train_ts.iloc[i][col].values)), mode='constant')
    for i in range(len(X_test_ts)):
        for col_idx, col in enumerate(X_test_ts.columns):
            X_test[i, col_idx] = np.pad(X_test_ts.iloc[i][col].values, pad_width=(0,max_len-len(X_test_ts.iloc[i][col].values)), mode='constant')
    return np.transpose(X_train, (0, 2, 1)), pd.Categorical(pd.Series(y_train)).codes,\
           np.transpose(X_test, (0, 2, 1)), pd.Categorical(pd.Series(y_test)).codes

