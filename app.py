import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from fredapi import Fred
import pandas_ta as ta
from datetime import datetime

# STREAMLIT CONFIG

st.set_page_config(
    page_title="S&P 500 Return Prediction",
    layout="wide"
)

st.title("📈 S&P 500 Daily Return Prediction")
st.write(
    "Forecasting daily **log returns** using a stacking ensemble of "
    "Linear Regression, Ridge Regression, and Decision Tree models."
)


# LOAD SECRETS 

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    st.error(
        "FRED_API_KEY not found in Streamlit secrets. "
        "Add it to `.streamlit/secrets.toml` or the Streamlit Cloud dashboard."
    )
    st.stop()

fred = Fred(api_key=FRED_API_KEY)


# SIDEBAR CONTROLS

st.sidebar.header("Configuration")

start_date = st.sidebar.date_input("Start Date", datetime(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()


# LOAD STACKED MODEL

@st.cache_resource
def load_model():
    with open("stack_model.pkl", "rb") as f:
        return pickle.load(f)

stack_model = load_model()


# LOAD PRICE DATA

@st.cache_data
def load_price_data(start, end):
    data = yf.download("^GSPC", start=start, end=end)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index)

    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    return data

price_data = load_price_data(start_date, end_date)


# LOAD MACRO DATA

@st.cache_data
def load_macro_data(start, end):
    ir = fred.get_series("FEDFUNDS", start, end)
    unemp = fred.get_series("UNRATE", start, end)
    cpi = fred.get_series("CPIAUCSL", start, end)

    macro = pd.concat([ir, unemp, cpi], axis=1)
    macro.columns = ["InterestRate", "Unemployment", "InflationRate"]
    macro.index = pd.to_datetime(macro.index)

    macro = macro.resample("D").ffill()
    return macro

macro_data = load_macro_data(start_date, end_date)


# MERGE DATA

df = price_data.join(macro_data, how="left")


# FEATURE ENGINEERING

df["SMA_20"] = ta.sma(df["Close"], length=20)
df["EMA_20"] = ta.ema(df["Close"], length=20)
df["RSI_14"] = ta.rsi(df["Close"], length=14)

macd = ta.macd(df["Close"])
df["MACD"] = macd.filter(like="MACD").iloc[:, 0]
df["Signal"] = macd.filter(like="MACDs").iloc[:, 0]

bb = ta.bbands(df["Close"], length=20)
df["BB_lower"] = bb.filter(like="BBL").iloc[:, 0]
df["BB_middle"] = bb.filter(like="BBM").iloc[:, 0]
df["BB_upper"] = bb.filter(like="BBU").iloc[:, 0]

df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)


# TARGET

df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

df = df.dropna()


# FEATURES (MUST MATCH TRAINING)

FEATURE_COLUMNS = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "InterestRate", "Unemployment", "InflationRate",
    "SMA_20", "EMA_20", "RSI_14",
    "MACD", "Signal",
    "BB_lower", "BB_middle", "BB_upper",
    "ATR"
]

X = df[FEATURE_COLUMNS].shift(1).dropna()
df = df.loc[X.index]


# PREDICTION

df["Predicted_Return"] = stack_model.predict(X)


# DISPLAY

st.subheader("📊 Prediction Results")

st.dataframe(
    df[["log_return", "Predicted_Return"]].tail(20),
    use_container_width=True
)

st.line_chart(df[["log_return", "Predicted_Return"]])

st.success("Prediction completed successfully.")

# FOOTER

st.markdown(
    """
    **Model:** Stacking Ensemble (Linear Regression, Ridge Regression, Decision Tree)  
    **Target:** Daily log returns  
    **Data Sources:** Yahoo Finance, FRED  
    **Purpose:** Academic research and demonstration
    """
)
