import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from fredapi import Fred
from dotenv import load_dotenv
import os
import pandas_ta as ta
from datetime import datetime

# ==================================================
# LOAD ENVIRONMENT VARIABLES (LOCAL TESTING)
# ==================================================
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

if FRED_API_KEY is None:
    st.error("FRED_API_KEY not found. Add it to your .env file.")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config(
    page_title="S&P 500 Return Prediction",
    layout="wide"
)

st.title("📈 S&P 500 Daily Return Prediction")
st.write(
    "Forecasting daily **log returns** using a stacking ensemble of "
    "Linear Regression, Ridge Regression, and Decision Tree models."
)

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
st.sidebar.header("Configuration")

start_date = st.sidebar.date_input("Start Date", datetime(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# ==================================================
# LOAD STACKED MODEL
# ==================================================
@st.cache_resource
def load_model():
    with open("stack_model.pkl", "rb") as f:
        return pickle.load(f)

stack_model = load_model()

# ==================================================
# LOAD PRICE DATA
# ==================================================
@st.cache_data
def load_price_data(start, end):
    data = yf.download("^GSPC", start=start, end=end)

    # Flatten MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index)

    # 🔴 CRITICAL: Ensure Adj Close exists (schema consistency)
    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    return data

price_data = load_price_data(start_date, end_date)

# ==================================================
# LOAD MACROECONOMIC DATA
# ==================================================
@st.cache_data
def load_macro_data(start, end):
    ir = fred.get_series("FEDFUNDS", start, end)
    unemp = fred.get_series("UNRATE", start, end)
    cpi = fred.get_series("CPIAUCSL", start, end)

    macro = pd.concat([ir, unemp, cpi], axis=1)
    macro.columns = ["InterestRate", "Unemployment", "InflationRate"]
    macro.index = pd.to_datetime(macro.index)

    # Monthly → Daily
    macro = macro.resample("D").ffill()

    return macro

macro_data = load_macro_data(start_date, end_date)

# ==================================================
# MERGE DATA
# ==================================================
df = price_data.join(macro_data, how="left")

# ==================================================
# FEATURE ENGINEERING (MATCHES TRAINING)
# ==================================================
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

# ==================================================
# TARGET VARIABLE (LOG RETURNS)
# ==================================================
df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

# ==================================================
# CLEAN DATA
# ==================================================
df = df.dropna()

# ==================================================
# FEATURE LIST (EXACT MATCH WITH X_train)
# ==================================================
FEATURE_COLUMNS = [
    "Adj Close", "Close", "High", "Low", "Open", "Volume",
    "InterestRate", "Unemployment", "InflationRate",
    "SMA_20", "EMA_20", "RSI_14",
    "MACD", "Signal",
    "BB_lower", "BB_middle", "BB_upper",
    "ATR"
]

# ==================================================
# BUILD MODEL INPUT (NO LOOK-AHEAD)
# ==================================================
X = df[FEATURE_COLUMNS].shift(1).dropna()
df = df.loc[X.index]

# 🔒 SAFETY CHECK (REMOVE AFTER TESTING IF YOU WANT)
assert list(X.columns) == FEATURE_COLUMNS, "Feature mismatch with training data"

# ==================================================
# PREDICTION
# ==================================================
df["Predicted_Return"] = stack_model.predict(X)

# ==================================================
# DISPLAY RESULTS
# ==================================================
st.subheader("📊 Prediction Results")

st.dataframe(
    df[["log_return", "Predicted_Return"]].tail(20),
    use_container_width=True
)

st.line_chart(df[["log_return", "Predicted_Return"]])

st.success("Prediction completed successfully.")

# ==================================================
# FOOTER
# ==================================================
st.markdown(
    """
    **Model:** Stacking Ensemble (Linear Regression, Ridge Regression, Decision Tree)  
    **Target:** Daily log returns  
    **Data Sources:** Yahoo Finance, FRED  
    **Purpose:** Academic research and demonstration
    """
)
