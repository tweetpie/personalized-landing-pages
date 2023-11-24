import pandas as pd

from pandas import DataFrame

from typing import List
from typing import Tuple

from sklearn.model_selection import train_test_split


def stratified_split(df: DataFrame, target_col: str, split_ratios: List[float]) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Splits a dataframe into train validation and test

    split_ratio :: [train, val, test]
    """
    train_size, val_size, test_size = split_ratios

    columns = [col for col in df.columns if col != target_col]
    print(columns)
    print(target_col)
    X, y = df[columns], df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=11)

    test_size_renormalized = test_size / (test_size + val_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=test_size_renormalized, stratify=y_test, random_state=11)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    return df_train, df_val, df_test


if __name__ == '__main__':

    df = pd.read_csv('/Users/raulferreira/pytorch-roberta-classifier/data/fake_news_sample.csv')

    train, val, test = stratified_split(df, 'label', split_ratios=[0.7, 0.2, 0.1])

