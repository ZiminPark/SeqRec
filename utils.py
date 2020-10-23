import pandas as pd


def get_vocab(df: pd.DataFrame, columns: str):
    return {index: item_id for item_id, index in enumerate(df[columns].unique())}
