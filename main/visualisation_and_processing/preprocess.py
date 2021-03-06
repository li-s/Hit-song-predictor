import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from collections import Counter


def preprocess_data(verbose=1):
    raw_df = pd.read_csv("../../data/data.csv")
    # drop less useful and intermediate value columns
    df_final = raw_df.drop(['artists', 'id', 'name', 'release_date',
                            'followers', 'year', 'popularity'], axis=1)

    if verbose:
        columns = df_final.columns
        print(f"These are the {len(columns)} columns of the trimmed dataset:")
        for i in range(0, len(columns)):
            print(f"{i + 1}: {columns[i]}")

        print(df_final.describe())

    return df_final


def split_data(verbose=1):
    df_final = preprocess_data(verbose)
    X = df_final.drop('is_in_billboard', axis=1)
    y = df_final['is_in_billboard']

    dataset_size = X.shape[0]
    training_size = round(0.8 * dataset_size)
    testing_size = dataset_size - training_size

    # split first before normalization, prevent data snooping
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
        X, y, train_size=training_size, test_size=testing_size)

    oversample = SMOTE(random_state=3)

    # Balance training dataset
    X_train_oversampled, y_train_oversampled = oversample.fit_resample(train_X, train_y)
    train_X = pd.DataFrame(X_train_oversampled, columns=X.columns)
    train_y = y_train_oversampled

    # Balance test dataset
    X_test_oversampled, y_test_oversampled = oversample.fit_resample(test_X, test_y)
    test_X = pd.DataFrame(X_test_oversampled, columns=X.columns)
    test_y = y_test_oversampled

    if verbose:
        # Check to see our new target column has a balanced number of positive and negative classes
        counter_train = Counter(train_y)
        counter_test = Counter(test_y)
        print("\nTraining dataset split:", counter_train)
        print("Test dataset split:", counter_test)
        print(
            f"Number of popular in train: {np.sum(train_y)}, Number of unpopular in train:{train_y.shape[0] - np.sum(train_y)}")
        print(f"Percentage of popular in train: {(np.sum(train_y) / (train_y.shape[0])) * 100}%\n")
        print(
            f"Number of popular in test: {np.sum(test_y)}, Number of unpopular in test:{test_y.shape[0] - np.sum(test_y)}")
        print(f"Percentage of popular in test: {((np.sum(test_y)) / (test_y.shape[0])) * 100}%")

    return train_X, train_y, test_X, test_y


def normalise():
    train_X, train_Y, test_X, test_Y = split_data(0)
    sc = sklearn.preprocessing.StandardScaler()
    normalized_train_X = pd.DataFrame(data=sc.fit_transform(train_X), index=[train_X.index], columns=[train_X.columns])
    normalized_test_X = pd.DataFrame(data=sc.transform(test_X), index=[test_X.index], columns=[test_X.columns])

    return normalized_train_X, train_Y, normalized_test_X, test_Y


if __name__ == "__main__":
    preprocess_data(1)
