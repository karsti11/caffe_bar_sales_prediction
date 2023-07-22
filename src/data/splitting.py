import pandas as pd
import itertools

def split_dataset(all_data_df: pd.DataFrame, 
                  test_split_date: str, 
                  dependent_var: str):
    """Split dataset by date. 
    First date of test is test_split_date.
    """
    X_train = all_data_df[all_data_df.index < test_split_date].drop(dependent_var, axis=1).copy()
    X_test = all_data_df[all_data_df.index >= test_split_date].drop(dependent_var, axis=1).copy()
    y_train = all_data_df[all_data_df.index < test_split_date][dependent_var].copy()
    y_test = all_data_df[all_data_df.index >= test_split_date][dependent_var].copy()
    print(f"Train dataset is from {X_train.index.min().strftime('%Y-%m-%d')} to {X_train.index.max().strftime('%Y-%m-%d')}")
    print(f"Test dataset is from {X_test.index.min().strftime('%Y-%m-%d')} to {X_test.index.max().strftime('%Y-%m-%d')}")
    return X_train, X_test, y_train, y_test

def time_series_cv(raw_data_filled_df, num_train_years, percentage_cut):
    """Custom time-series split in train-validation sets per year.
    If there are more than 3 years in training dataset:
    First split: training dataset is first 3 years of data, validation set 4th year data
    Second split: training dataset is first 4 years of data, validation set 5th year data
    ... and so on until last year of train dataset is set as validation set.

    :param num_train_years: number of years in training dataset
    :param percentage_cut: which percentage of dataset to use for training when there
                            is not much data
    """
    groups = raw_data_filled_df.reset_index().groupby(raw_data_filled_df.index.year).groups
    sorted_groups = [value.tolist() for (key, value) in sorted(groups.items())]#list of indices per year
    print(f"Groups: {groups.keys()}")
    if len(groups.keys()) < num_train_years:
        cut_idx = int(sorted_groups[-1][-1]*percentage_cut) # First validation set cut
        val_cut_idx = int(sorted_groups[-1][-1]*(percentage_cut+(1-percentage_cut)/2)) # Second validation set cut
        all_indices = list(itertools.chain(*sorted_groups))
        return [(all_indices[:cut_idx], all_indices[cut_idx:val_cut_idx]), 
                (all_indices[:val_cut_idx], all_indices[val_cut_idx:])]
    elif len(groups.keys()) in (num_train_years, num_train_years+1):
        return [(list(itertools.chain(*sorted_groups[:-1])), sorted_groups[-1])]
    else:
        return [(list(itertools.chain(*sorted_groups[i:num_train_years+i])), sorted_groups[i+num_train_years])
          for i in range(len(sorted_groups) - num_train_years)]

def check_validation_splits_sparsity(cv_splits_indices, y_train):
    """Prints out percentage of non-sales days.
    """
    for num, idxs in enumerate(cv_splits_indices):
        print(f"Time series validaton split {num} ...")
        train_set = y_train[y_train.index[idxs[0]]]
        valid_set = y_train[y_train.index[idxs[1]]]
        train_set_sparsity = (len(train_set[train_set > 0]) / len(train_set))*100.0
        valid_set_sparsity = (len(valid_set[valid_set > 0]) / len(valid_set))*100.0
        print("Train set sparsity: ", round(100 - train_set_sparsity,2))
        print("Validation set sparsity: ", round(100 - valid_set_sparsity, 2))

