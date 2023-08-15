import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from src.utils import get_project_root
from src.evaluation.scoring import wmape, wbias


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


def calculate_scores_per_item_last_365d(sales_and_predictions_df, current_date):
    # Calculate scores for last 365 days
    c1 = (sales_and_predictions_df.index >= str(current_date - datetime.timedelta(days=365)))
    c2 = (sales_and_predictions_df.index <= str(current_date - datetime.timedelta(days=1)))
    scores_df = sales_and_predictions_df[c1 & c2].groupby('item_name').apply(
        lambda x: pd.Series({
            'bias': wbias(x['sales_qty'], x['prediction']), 
            'wmape': wmape(x['sales_qty'], x['prediction']),
            'total_sales': x['sales_qty'].sum(),
            'total_prediction': x['prediction'].sum()})
        ).sort_values(by='total_sales', ascending=False).reset_index()

    return scores_df


def visualize_totals(scores_df):
    # Create the figure
    fig = go.Figure()
    X_axis = np.arange(len(scores_df['item_name']))
    # Create bar traces for 'sales' and 'prediction'
    fig.add_trace(go.Bar(
        x=X_axis - 0.2,
        y=scores_df['total_sales'],
        width=0.4,
        marker=dict(color='rgba(0, 0, 255, 0.5)'),
        name='sales'
    ))
    fig.add_trace(go.Bar(
        x=X_axis + 0.2,
        y=scores_df['total_prediction'],
        width=0.4,
        marker=dict(color='rgba(0, 128, 0, 0.5)'),
        name='prediction'
    ))
    # Update x-axis
    fig.update_xaxes(
        tickvals=X_axis,
        ticktext=scores_df['item_name'],
        tickangle=90,
        tickfont=dict(size=16)
    )
    # Create the first y-axis (for 'sales' and 'prediction')
    fig.update_yaxes(
        title_text='Sales & Prediction',
        title_font=dict(size=16),
        tickfont=dict(size=16),
        showgrid=False,
        zeroline=False
    )
    # Create the second y-axis (for 'wmape')
    fig.update_layout(
        yaxis2=dict(
            title_text='wmape',
            title_font=dict(size=16),
            tickfont=dict(size=16),
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
        )
    )
    # Create the third y-axis (for 'bias')
    fig.update_layout(
        yaxis3=dict(
            title_text='Bias',
            title_font=dict(size=16),
            tickfont=dict(size=16),
            overlaying='y',
            side='right',
            anchor='free',
            showgrid=False,
            zeroline=True,
            position=1.0
        ),
    )
    # Create line trace for 'wmape'
    fig.add_trace(go.Scatter(
        x=X_axis,
        y=scores_df['wmape'],
        mode='markers+lines',
        marker=dict(color='red', symbol='circle-open'),
        name='wmape',
        yaxis='y2'
    ))
    # Create line trace for 'bias'
    fig.add_trace(go.Scatter(
        x=X_axis,
        y=scores_df['bias'],
        mode='markers+lines',
        marker=dict(color='purple', symbol='triangle-up'),
        name='bias',
        yaxis='y3'
    ))
    # Update legend and layout
    fig.update_layout(
        xaxis=dict(
            domain=[0, 0.9]
        ),
        legend=dict(
            x=0.5,
            y=1,
            xanchor='center',
            font=dict(size=16)
        ),
        height=600,
        width=1200
    )
    st.plotly_chart(fig, theme="streamlit")



dataset_with_predictions = pd.read_pickle(WHOLE_DATASET_PATH)
dataset_with_inventory = pd.read_pickle(INVENTORY_DATASET_PATH)
all_items = dataset_with_predictions.item_name.unique().tolist()

st.set_page_config(layout="wide")

st.title('Model evaluation')

with st.sidebar:
    
    st.title(':female-scientist: Model evaluation :male-scientist:')
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

  
scores_df = calculate_scores_per_item_last_365d(dataset_with_predictions, current_date)
visualize_totals(scores_df)

option = st.selectbox('Select item to visualize predictions.', items_list)
if option:
    with st.container():
        visualize_preds(dataset_with_predictions, option, date_from, current_date, selected_date)
