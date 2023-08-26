import streamlit as st
from src.utils import get_project_root


st.set_page_config(layout="wide")

st.title(':coffee: Caffe bar sales, predictions and inventory analysis :wine_glass:')
st.write('')
st.subheader('Description')
st.write("""Hi! This app should help you have a better view of how your beverages retail business is doing.  
    It consists of two pages, first one is Main Overview which serves as a summary of inventory and predictions  
    for following dates and historical sales contribution by item. Second one is Model Evaluation, page for detailed analysis  
    of model predictions.""")

st.subheader('Content')
st.markdown('1. Main Overview')
st.markdown('2. Model Evaluation')




