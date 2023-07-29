import os
import time
import pandas as pd
from src.utils import get_project_root
from src.data.item_names_replacement import REPLACE_DICT1, REPLACE_DICT1

YEARS = [str(x) for x in list(range(2013,2021))]
ROOT_DIR = get_project_root()


def string_to_float(number):
    #Custom function for converting 'sales_value' column to float 
    #because of faulty data. 28 rows have eg. '400.200.000.000.000.000'
    try:
        return float(number)
    except:
        return 0.5

def load_data(data_abs_path: str) -> pd.DataFrame:
    """Load raw data
    
    Parameters:
    -----------
    data_abs_path: absolute path of csv data

    Returns:
    --------
    data_df: raw data dataframe

    """
    data_df = pd.read_csv(data_abs_path)
    data_df.sales_datetime = pd.to_datetime(data_df.sales_datetime, format='%Y-%m-%d', utc=True)
    data_df.set_index('sales_datetime', inplace=True)
    return data_df

def arrange_data(data_df):
    # Drop unnecessary columns -> no known meaning
    data_df.drop(labels=[4,10,11], axis=1, inplace=True)
    data_df.columns = ['bar_name', 'number2', 'feature1', 'sales_datetime', 'feature2', 
                          'item_name', 'item_class', 'sales_qty', 'feature3', 'sales_value']
    #data_df.sales_value=data_df.sales_value.apply(lambda x: string_to_float(x))
    data_df.sales_datetime = pd.to_datetime(data_df.sales_datetime, format='mixed', utc=True)
    data_df.set_index('sales_datetime', inplace=True)
    data_df['item_price'] = abs(data_df['sales_value']/data_df['sales_qty'])
    return data_df

def load_dataset():

    columns_to_keep = ['item_name', 'sales_qty', 'sales_value', 'item_price']
    all_data_df = pd.DataFrame(columns = columns_to_keep)
    for year in YEARS:
        start_time = time.time()
        filename = os.path.join(ROOT_DIR, f'data/raw/{year}_eKasa_RECEIPT_ENTRIES.csv') 
        df = pd.read_csv(filename, 
                         delimiter=';', 
                         header=None,
                         converters={12: string_to_float},
                         encoding='latin-1')
        data_df = arrange_data(df)
        all_data_df = pd.concat([all_data_df, data_df[columns_to_keep]])
        print("Dataframe shape: ",df.shape)
        #print("Dataframe head: ",df.head())
        end_time = time.time()
        print("Time (s): ", end_time-start_time)
        print(f"{year} done.")
    all_data_df.sales_qty = all_data_df.sales_qty.astype('int64')
    all_data_df.item_name.replace(to_replace=REPLACE_DICT1, inplace=True)
    all_data_df.item_name.replace(to_replace=REPLACE_DICT1, inplace=True)
    all_data_df.index.name = 'sales_date'
    all_data_daily_sales = all_data_df.groupby(['item_name', pd.Grouper(freq='D')]).agg({'sales_qty':'sum', 
                                                                                          'item_price': 'mean', 
                                                                                         'sales_value': 'sum'}).reset_index()
    print(all_data_daily_sales.head())

    return all_data_daily_sales


def load_dataset_debug():

    columns_to_keep = ['item_name', 'sales_qty', 'sales_value', 'item_price']
    all_data_df = pd.DataFrame(columns = columns_to_keep)
    for year in YEARS:
        start_time = time.time()
        filename = os.path.join(ROOT_DIR, f'data/raw/{year}_eKasa_RECEIPT_ENTRIES.csv') 
        df = pd.read_csv(filename, 
                         delimiter=';', 
                         header=None,
                        converters={12: string_to_float})
        data_df = arrange_data(df)
        all_data_df = all_data_df.append(data_df[columns_to_keep])
        print("Dataframe shape: ",df.shape)
        #print("Dataframe head: ",df.head())
        end_time = time.time()
        print("Time (s): ", end_time-start_time)
        print(f"{year} done.")
    all_data_df.sales_qty = all_data_df.sales_qty.astype('int64')
    all_data_df.item_name.replace(to_replace=REPLACE_DICT1, inplace=True)
    all_data_df.item_name.replace(to_replace=REPLACE_DICT1, inplace=True)
    all_data_df.index.name = 'sales_date'
    #all_data_daily_sales = all_data_df.groupby(['item_name', pd.Grouper(freq='D')]).agg({'sales_qty':'sum', 
    #                                                                                      'item_price': 'mean', 
    #                                                                                     'sales_value': 'sum'}).reset_index()
    #print(all_data_daily_sales)

    return all_data_df