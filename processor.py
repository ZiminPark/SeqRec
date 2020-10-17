import numpy as np
import pandas as pd
import datetime as dt

from pathlib import Path


def load_data(data_path: Path, nrows=None):
    data = pd.read_csv(data_path / 'yoochoose-clicks.dat', sep=',', header=None, usecols=[0, 1, 2],
                       parse_dates=[1], dtype={0: np.int32, 2: np.int32}, nrows=nrows)
    data.columns = ['SessionId', 'Time', 'ItemId']
    return data


def cleanse_minor(data: pd.DataFrame, shortest=2, least_click=5) -> pd.DataFrame:
    while True:
        before_len = len(data)

        session_len = data.groupby('SessionId').size()
        # noinspection PyTypeChecker
        session_use = session_len[session_len >= shortest].index
        data = data[data['SessionId'].isin(session_use)]

        item_popular = data.groupby('ItemId').size()
        item_use = item_popular[item_popular >= least_click].index
        data = data[data['ItemId'].isin(item_use)]

        after_len = len(data)
        if before_len == after_len:
            break
    return data


def split_by_date(data: pd.DataFrame, n_days: int):
    final_time = data['Time'].max()
    session_last_time = data.groupby('SessionId')['Time'].max()
    session_in_train = session_last_time[session_last_time < final_time - dt.timedelta(n_days)].index
    session_in_test = session_last_time[session_last_time >= final_time - dt.timedelta(n_days)].index

    before_date = data[data['SessionId'].isin(session_in_train)]
    after_date = data[data['SessionId'].isin(session_in_test)]
    after_date = after_date[after_date['ItemId'].isin(before_date['ItemId'])]
    return before_date, after_date


test_length = test.groupby('SessionId').size()
test = test[np.in1d(test.SessionId, test_length[test_length >= 2].index)]

print(
    f'Full train set\n\tEvents: {len(train)}\n\tSessions: {train.SessionId.nunique()}\n\tItems: {train.ItemId.nunique()}')
train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_full.txt', sep='\t', index=False)

print(f'Test set\n\tEvents: {len(test)}\n\tSessions: {test.SessionId.nunique()}\n\tItems: {test.ItemId.nunique()}')
test.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_test.txt', sep='\t', index=False)

max_train_time = train.Time.max()
session_max_times = train.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < max_train_time - dt.timedelta(1)].index
session_valid = session_max_times[session_max_times >= max_train_time - dt.timedelta(1)].index
train_tr = train[np.in1d(train.SessionId, session_train)]
valid = train[np.in1d(train.SessionId, session_valid)]
valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
valid_length = valid.groupby('SessionId').size()
valid = valid[np.in1d(valid.SessionId, valid_length[valid_length >= 2].index)]
print(
    f'Train set\n\tEvents: {len(train_tr)}\n\tSessions: {train_tr.SessionId.nunique()}\n\tItems: {train_tr.ItemId.nunique()}')
train_tr.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_tr.txt', sep='\t', index=False)

print(
    f'Validation set\n\tEvents: {len(valid)}\n\tSessions: {valid.SessionId.nunique()}\n\tItems: {valid.ItemId.nunique()}')
valid.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_valid.txt', sep='\t', index=False)
