import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from src.utils import get_project_root
from src.evaluation.scoring import wbias, wmape, bias
from src.streamlit_app.helper_functions import load_dataset


DATE_FROM = datetime.date(2019, 1, 1)
DATE_TO = datetime.date(2019, 12, 31)

DATASETS_FOLDER = get_project_root() / 'data/processed'
WHOLE_DATASET_PATH = DATASETS_FOLDER / 'dataset_with_predictions.pkl'
INVENTORY_DATASET_PATH = DATASETS_FOLDER / 'inventory_data_top40.pkl'


def get_inventory_on_current_date(inventory_df, items_list, selected_date):
    c1 = (inventory_df.item_name.isin(items_list))
    c2 = (inventory_df.index == str(selected_date))

    return inventory_df[c1 & c2].groupby('item_name')['inventory'].sum().reset_index()


def get_aggregated_predictions(predictions_df, items_list, date_from, date_to):
    c1 = (predictions_df.item_name.isin(items_list))
    c2 = (predictions_df.index >= str(date_from))
    c3 = (predictions_df.index <= str(date_to))
    predictions_by_item = predictions_df[c1 & c2 & c3].groupby('item_name')['prediction'].sum().reset_index() 
    predictions_by_item['prediction'] = predictions_by_item['prediction'].round().astype('int')

    return predictions_by_item  


def get_last_365d_sales(sales_df, items_list, current_date):
    c1 = (sales_df.item_name.isin(items_list))
    c2 = (sales_df.index >= str(current_date - datetime.timedelta(days=365)))
    c3 = (sales_df.index <= str(current_date - datetime.timedelta(days=1)))
    sales_by_item = sales_df[c1 & c2 & c3].groupby('item_name')['sales_qty'].sum().reset_index().sort_values(by='sales_qty', ascending=False)

    return sales_by_item



class KPIsCalculation:

    def __init__(self, current_date):
        self.current_date = current_date
        self.current_year = self.current_date.year
        self.last_year = self.current_year - 1
        self.year_ago_date = self.current_date - relativedelta(years=1)


    def get_yoy_sales(self, sales_df):
        # Calculate total sales YTD and compare with last year, same period

        c1 = (sales_df.index.year == self.last_year)
        c2 = (sales_df.index < str(self.year_ago_date)) 
        c3 = (sales_df.index.year == self.current_year)
        c4 = (sales_df.index < str(self.current_date))

        last_year_totals = sales_df[c1 & c2][['sales_qty', 'sales_value']].sum()
        this_year_totals = sales_df[c3 & c4][['sales_qty', 'sales_value']].sum()

        percent_change_qty = -(1 - (this_year_totals['sales_qty']/last_year_totals['sales_qty'])).round(2)*100
        percent_change_val = -(1 - (this_year_totals['sales_value']/last_year_totals['sales_value'])).round(2)*100

        return (
            this_year_totals['sales_qty'],
            this_year_totals['sales_value'],
            percent_change_qty,
            percent_change_val
            )

     
    def get_yoy_predictions(self, predictions_df):
        # Predictions
        c1 = (predictions_df.index.year == self.last_year)
        c2 = (predictions_df.index < str(self.year_ago_date)) 
        c3 = (predictions_df.index.year == self.current_year)
        c4 = (predictions_df.index < str(self.current_date))

        last_year_preds = predictions_df[c1 & c2][['sales_qty', 'prediction']]
        this_year_preds = predictions_df[c3 & c4][['sales_qty', 'prediction']]

        last_year_wmape = wmape(last_year_preds['sales_qty'], last_year_preds['prediction'])
        this_year_wmape = wmape(this_year_preds['sales_qty'], this_year_preds['prediction'])

        last_year_bias = bias(last_year_preds['sales_qty'], last_year_preds['prediction'])
        this_year_bias = bias(this_year_preds['sales_qty'], this_year_preds['prediction'])

        wmape_pp_change = last_year_wmape - this_year_wmape
        bias_pp_change = abs(last_year_bias) - abs(this_year_bias)

        return (
            wmape_pp_change,
            bias_pp_change,
            this_year_wmape,
            this_year_bias
            )


    def get_yoy_inventory(self, inventory_df):    

        # Inventory
        c1 = (inventory_df.index.year == self.last_year)
        c2 = (inventory_df.index < str(self.year_ago_date)) 
        c3 = (inventory_df.index.year == self.current_year)
        c4 = (inventory_df.index < str(self.current_date))

        last_year_inv = inventory_df[c1 & c2][['sales_qty', 'inventory']]
        this_year_inv = inventory_df[c3 & c4][['sales_qty', 'inventory']]  

        last_year_oos_cases= len(last_year_inv[(last_year_inv.inventory < last_year_inv.sales_qty)])
        this_year_oos_cases= len(this_year_inv[(this_year_inv.inventory < this_year_inv.sales_qty)])

        percent_change_cases = round(-(1 - (this_year_oos_cases/last_year_oos_cases))*100, 1)

        return (
            percent_change_cases,
            this_year_oos_cases
            ) 


def visualize_last_365d_sales(sales_df):    
    fig = px.bar(
        sales_df, 
        x='item_name', 
        y='sales_qty',
        labels={'sales_qty': 'Total sold quantity', 'item_name':'Item name'},
        height=600,
        width=1200)
    st.plotly_chart(fig, theme="streamlit")


st.set_page_config(layout="wide")

dataset_with_predictions = load_dataset(WHOLE_DATASET_PATH)
dataset_with_inventory = load_dataset(INVENTORY_DATASET_PATH)
all_items = dataset_with_predictions.item_name.unique().tolist()
# Sidebar
## Title
with st.sidebar:
    st.title(':chart_with_upwards_trend: Main Overview :chart_with_upwards_trend:')
    st.subheader('1. Select items')
    items_list = st.multiselect('Which items you want inventory and predictions summary for?', all_items, default=all_items)
    st.subheader('2. Select current date')
    current_date = st.date_input("Current date:", datetime.date(2019, 3, 1))
    st.subheader('3. Select last date in future')
    selected_date = st.slider(
             "Select days from current_date for future summary",
             min_value=current_date,
             max_value=current_date + datetime.timedelta(days=90),
             value=current_date + datetime.timedelta(days=7),
             format="DD/MM/YYYY")
    ## Select dates header


#with col2:
    # Main screen

st.header(f'1. How is our bussines doing? (KPIs Overview)')
st.write("How is our bussines doing?")
kpis_calculation = KPIsCalculation(current_date)

this_year_sales_qty, this_year_sales_val, percent_change_qty, percent_change_val = kpis_calculation.get_yoy_sales(dataset_with_predictions)
wmape_pp_change, bias_pp_change, this_year_wmape, this_year_bias = kpis_calculation.get_yoy_predictions(dataset_with_predictions)
percent_change_cases, this_year_oos_cases = kpis_calculation.get_yoy_inventory(dataset_with_inventory)

col1, col2, col3 = st.columns(3)
col1.metric("Sales quantity (YTD)", f"{this_year_sales_qty} pcs", f"{percent_change_qty} %")
col1.metric("Sales value (YTD)", f"{this_year_sales_val} Kn", f"{percent_change_val} %")
col2.metric("Forecast error (YTD - on average)", f"{this_year_wmape} %", f"{wmape_pp_change} pp")
col2.metric("Forecast bias in total (YTD)", f"{this_year_bias} %", f"{bias_pp_change} pp")
col3.metric("Out of Stock situations", f"{this_year_oos_cases} cases", f"{percent_change_cases} %")



st.header(f'2. What is the current and predicted inventory status? (Inventory on current date up to {selected_date})')

inventory_on_current_date = get_inventory_on_current_date(dataset_with_inventory, items_list, current_date)
aggregated_predictions = get_aggregated_predictions(dataset_with_predictions, items_list, current_date, selected_date).round(1)
inventory_and_predictions = inventory_on_current_date.merge(aggregated_predictions, how='left', on=['item_name'])
inventory_and_predictions.loc[:, 'inventory_at_selected_date'] = inventory_and_predictions['inventory'] - inventory_and_predictions['prediction']
st.dataframe(
    inventory_and_predictions.sort_values(by='inventory_at_selected_date'), 
    column_config={
        'item_name': 'Item name', 
        'inventory': 'Inventory on selected (current) date',
        'prediction': f'Total predicted sales ({current_date} to {selected_date})',
        'inventory_at_selected_date': 'Inventory at selected date (End of day)'},
    hide_index=True)

st.header(f'3. What are the top selling items? (Total sales quantity in 365 days before {current_date})')
sales_last_365d = get_last_365d_sales(dataset_with_predictions, all_items, current_date)
visualize_last_365d_sales(sales_last_365d)

