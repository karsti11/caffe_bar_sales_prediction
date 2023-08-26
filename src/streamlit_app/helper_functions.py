import xgboost
import pandas as pd
import streamlit as st


@st.cache_data
def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_pickle(dataset_path)


@st.cache_resource
def load_booster(booster_path: str):
    booster = xgboost.Booster()
    booster.load_model(booster_path)
    if booster.attr('feature_names') is not None: 
        booster.feature_names = booster.attr('feature_names').split('|')  
    return booster