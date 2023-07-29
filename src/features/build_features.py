import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.calendar import easter_dates, easter_monday_dates

def fill_time_series(
    raw_data_df: pd.DataFrame
    ) -> pd.DataFrame:
    """Fills data for missing dates in raw dataframe per item. 
    Dataframe must have daily DatetimeIndex.
    """
    data_df = raw_data_df.resample('D').sum()
    data_df.loc[:, 'item_price'] = data_df.item_price.replace(to_replace=0, method='ffill')
    return data_df


class MetadataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop
    def fit(self, X, y=None):
        self.columns_to_drop = ['sales_value']
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        columns_to_keep = [col for col in X_.columns if col not in self.columns_to_drop]
        X_ = X_[columns_to_keep]
        return X_

class CalendarTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_daysofweek=True):
        self.add_daysofweek = add_daysofweek

    def fit(self, X, y=None):
        self.min_year = X.index.year.min()
        return self

    def transform(self, X, y=None):
        X_ = X.copy() # creating a copy to avoid changes to original dataset
        #X_.loc[:,'day_of_week'] = X_.index.day_of_week
        if not isinstance(X_.index, pd.DatetimeIndex):
            X_ = X_.set_index('sales_date')
        if self.add_daysofweek:
            for day in range(1,7):
                X_.loc[:, f'day_of_week_{day}'] = (X_.index.day_of_week == day).astype('int8')
        for month in range(1,12):
            X_.loc[:, f'month_of_year_{month}'] = (X_.index.month == month).astype('int8')
        # X_.index.year.min() should be generated during fit 
        X_.loc[:,'year'] = X_.index.year - self.min_year
        X_.loc[:,'first_third_of_month'] = (X_.index.day <= 10).astype('int8')
        X_.loc[:,'second_third_of_month'] = ((X_.index.day > 10) & (X_.index.day <= 20)).astype('int8')
        X_.loc[:,'last_third_of_month'] = (X_.index.day > 20).astype('int8')
        # Temporarily closed because of COVID pandemic
        X_.loc[:,'closed'] = ((X_.index >= '2020-03-19') & (X_.index >= '2020-05-10')).astype('int8')


        return X_


class HolidaysTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy() # creating a copy to avoid changes to original dataset
        X_.loc[:, 'easter'] = X_.index.isin(easter_dates).astype('int8')
        X_.loc[:, 'easter_monday'] = X_.index.isin(easter_monday_dates).astype('int8')
        X_.loc[:, 'christmas'] = ((X_.index.month==12) & (X_.index.day==25)).astype('int8')
        X_.loc[:, 'new_years_day'] = ((X_.index.month==1) & (X_.index.day==1)).astype('int8')
        X_.loc[:, 'new_years_eve'] = ((X_.index.month==12) & (X_.index.day==31)).astype('int8')
        X_.loc[:, 'sv_lovre'] = ((X_.index.month==8) & (X_.index.day==10)).astype('int8')
        X_.loc[:, 'prvi_maj'] = ((X_.index.month==5) & (X_.index.day==1)).astype('int8')
        self.feature_names = X_.columns.tolist()

        return X_

    def get_feature_names(self):
        return self.feature_names



def add_calendar_features(
    raw_data_df: pd.DataFrame
    ) -> pd.DataFrame:

    """Adds calendar features to raw dataframe(days of week, month of year, 
    year, thirds of month). Dataframe must have daily DatetimeIndex.
    """
    transformed_data_df = raw_data_df.copy()
    transformed_data_df.loc[:,'day_of_week'] = transformed_data_df.index.day_of_week
    transformed_data_df.loc[:,'month_of_year'] = transformed_data_df.index.month
    day_of_week_dummies = pd.get_dummies(transformed_data_df.day_of_week, prefix='day_of_week', drop_first=True)
    month_of_year_dummies = pd.get_dummies(transformed_data_df.month_of_year, prefix='month_of_year', drop_first=True)
    transformed_data_df = transformed_data_df.merge(day_of_week_dummies, how='left', left_index=True, right_index=True)
    transformed_data_df = transformed_data_df.merge(month_of_year_dummies, how='left', left_index=True, right_index=True)
    transformed_data_df.loc[:,'year'] = transformed_data_df.index.year - transformed_data_df.index.year.min()
    transformed_data_df.loc[:,'first_third_of_month'] = (transformed_data_df.index.day <= 10).astype('int8')
    transformed_data_df.loc[:,'second_third_of_month'] = ((transformed_data_df.index.day > 10) & (transformed_data_df.index.day <= 20)).astype('int8')
    transformed_data_df.loc[:,'last_third_of_month'] = (transformed_data_df.index.day > 20).astype('int8')
    transformed_data_df.drop(columns=['day_of_week', 'month_of_year'], inplace=True)

    return transformed_data_df

def add_holidays_features(
    data_df: pd.DataFrame
    ) -> pd.DataFrame:
    """Add easter and easter monday dummy variables. 
    DataFrame must have DatetimeIndex.
    """

    data_df.loc[:, 'easter'] = data_df.index.isin(easter_dates).astype('int8')
    data_df.loc[:, 'easter_monday'] = data_df.index.isin(easter_monday_dates).astype('int8')
    data_df.loc[:, 'christmas'] = ((data_df.index.month==12) & (data_df.index.day==25)).astype('int8')
    data_df.loc[:, 'new_years_day'] = ((data_df.index.month==1) & (data_df.index.day==1)).astype('int8')
    data_df.loc[:, 'new_years_eve'] = ((data_df.index.month==12) & (data_df.index.day==31)).astype('int8')
    
    return data_df

#def add_special_events(data_df: pd.DataFrame) -> pd.DataFrame:
