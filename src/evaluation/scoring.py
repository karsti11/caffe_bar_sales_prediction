import numpy as np
import pandas as pd



def wmape(actual: pd.Series, forecast: pd.Series):
    """Weighted mean absolute percentage error:
        "...variant of MAPE in which errors are weighted by values of actuals..."
        from https://en.wikipedia.org/wiki/WMAPE

    Parameters:
    -----------
    actual: array of actual values
    forecast: array of forecasted values

    Returns:
    ----------
    score: Weighted MAPE
    """
    numerator = np.abs(actual - forecast).sum()
    denominator = actual.sum()
    if denominator == 0:
        denominator = 1.0
    score = (numerator/denominator)*100
    return round(score, 2)


def wbias(actual: pd.Series, forecast: pd.Series):
    """Weighted bias, positive value indicates forecast overshooting,
    negative values indicate forecast undershooting.

    Parameters:
    -----------
    actual: array of actual values
    forecast: array of forecasted values

    Returns:
    ----------
    score: Weighted Bias
    """
    numerator = (forecast - actual).sum()
    denominator = actual.sum()
    if denominator == 0:
        denominator = 1.0
    score = (numerator/denominator)*100
    return round(score, 2)


def bias(actual: pd.Series, forecast: pd.Series):

    return (1 - (actual.sum() / forecast.sum()))*100


def calculate_errors(y_train: pd.Series, 
                    y_test: pd.Series, 
                    y_pred_train: pd.Series, 
                    y_pred_test: pd.Series):

    train_wmape = wmape(y_train, y_pred_train)
    test_wmape = wmape(y_test, y_pred_test)
    train_wbias = wbias(y_train, y_pred_train)
    test_wbias = wbias(y_test, y_pred_test)
    print(f"Train WMAPE: {train_wmape}")   
    print(f"Test WMAPE: {test_wmape}")
    print(f"Train Wbias: {train_wbias}")   
    print(f"Test Wbias: {test_wbias}")
    return {
        "train_wmape": train_wmape,
        "test_wmape": test_wmape,
        "train_wbias": train_wbias,
        "test_wbias": test_wbias
    }




