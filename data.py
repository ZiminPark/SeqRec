import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = 'D:\\data\\yoochoose-data\\'
PATH_TO_PROCESSED_DATA = 'D:\\data\\yoochoose-data\\'

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'yoochoose-clicks.dat', sep=',', header=None, usecols=[0, 1, 2],
                   parse_dates=[1],
                   dtype={0: np.int32, 2: np.int32}, nrows=100000)
data.columns = ['SessionId', 'Time', 'ItemId']

session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]

item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]

session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]

tmax = data['Time'].max()
session_max_times = data.groupby('SessionId')['Time'].max()
session_train = session_max_times[session_max_times < tmax - dt.timedelta(1)].index
session_test = session_max_times[session_max_times >= tmax - dt.timedelta(1)].index

train = data[np.in1d(data.SessionId, session_train)]
test = data[np.in1d(data.SessionId, session_test)]

test = test[np.in1d(test.ItemId, train.ItemId)]

test_length = test.groupby('SessionId').size()
test = test[np.in1d(test.SessionId, test_length[test_length >= 2].index)]

print(
    f'Full train set\n\tEvents: {len(train)}\n\tSessions: {train.SessionId.nunique()}\n\tItems: {train.ItemId.nunique()}')
train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_full.txt', sep='\t', index=False)

print(f'Test set\n\tEvents: {len(test)}\n\tSessions: {test.SessionId.nunique()}\n\tItems: {test.ItemId.nunique()}')
test.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_test.txt', sep='\t', index=False)
