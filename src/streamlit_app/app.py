import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from src.utils import get_project_root


DATE_FROM = datetime.date(2019, 1, 1)
DATE_TO = datetime.date(2019, 12, 31)

DATASETS_FOLDER = get_project_root() / 'data/processed'
WHOLE_DATASET_PATH = DATASETS_FOLDER / 'dataset_with_predictions.pkl'
INVENTORY_DATASET_PATH = DATASETS_FOLDER / 'inventory_data_top40.pkl'

def visualize_preds(predictions_df, item_name, date_from, current_date, date_to):
    c1 = (predictions_df.item_name == item_name)
    c2 = (predictions_df.index >= str(date_from))
    c3 = (predictions_df.index <= str(date_to))
    preds_visualize = predictions_df[c1 & c2 & c3].copy()
    preds_visualize.loc[(preds_visualize.index >= str(current_date)), 'sales_qty'] = np.nan
    fig = px.line(preds_visualize,
                y=['sales_qty', 'prediction'],
                #barmode="group",
                labels={'sales_qty': 'Sold quantity', 'prediction':'Predicted daily quantities'},
                title=f'Sales and predictions for {item_name} from {date_from} to {date_to}',
                color_discrete_map={'sales_qty': 'blue', 'prediction': 'red'},
                line_dash_map = {'sales_qty': 'solid', 'prediction': 'dash'},
                markers=True,
                height=600,
                width=1200
                )
    # Change the bar mode
    # fig = plt.figure(figsize=(16, 9))
    # plt.bar(preds_visualize.index, preds_visualize.sales_qty)
    # plt.bar(preds_visualize.index, preds_visualize.prediction)
    st.plotly_chart(fig, theme="streamlit")


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


dataset_with_predictions = pd.read_pickle(WHOLE_DATASET_PATH)
dataset_with_inventory = pd.read_pickle(INVENTORY_DATASET_PATH)
all_items = dataset_with_predictions.item_name.unique().tolist()

st.set_page_config(layout="wide")

# Sidebar
## Title
with st.sidebar:
    st.title('Caffe bar sales')

    items_list = st.multiselect('Which items you want to explore?', all_items, default=all_items[:10])

    st.header('Select current date for future summary.')
    current_date = st.date_input("Current date:", datetime.date(2019, 3, 1))
    selected_date = st.slider(
             "Select days from current_date for future summary",
             min_value=DATE_FROM,
             max_value=current_date + datetime.timedelta(days=90),
             value=current_date,
             format="DD/MM/YYYY")
    ## Select dates header
    st.subheader('Select dates for sales forecast.')
    ## Input date range for predictions
    date_from = st.date_input("From date:", DATE_FROM)
    date_to = st.date_input("To date:", DATE_TO)
    #st.write(f"Generation of predictions from {date_from} to {date_to}")
    # Select items header
    st.subheader('Select items for sales forecast.')

#with col2:
    # Main screen
st.subheader(f'Inventory on current date and total predicted sales up to {selected_date}')

inventory_on_current_date = get_inventory_on_current_date(dataset_with_inventory, items_list, current_date)
aggregated_predictions = get_aggregated_predictions(dataset_with_predictions, items_list, current_date, selected_date).round(1)
inventory_and_predictions = inventory_on_current_date.merge(aggregated_predictions, how='left', on=['item_name'])
inventory_and_predictions.loc[:, 'inventory_at_selected_date'] = inventory_and_predictions['inventory'] - inventory_and_predictions['prediction']
st.dataframe(
    inventory_and_predictions, 
    column_config={
        'item_name': 'Item name', 
        'inventory': 'Inventory on selected (current) date',
        'prediction': f'Total predicted sales ({current_date} to {selected_date})',
        'inventory_at_selected_date': 'Inventory at selected date (End of day)'},
    hide_index=True)



    
option = st.selectbox('Select item to visualize predictions.', items_list)
if option:
    with st.container():
        visualize_preds(dataset_with_predictions, option, date_from, current_date, selected_date)
