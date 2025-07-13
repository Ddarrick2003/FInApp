import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var

from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="📊 Forecasting App", layout="wide")
st.title("📊 Financial Forecasting Dashboard (LSTM + GARCH)")

uploaded_file = st.file_uploader("📤 Upload your OHLCV CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
                # Clean column names
        df.columns = df.columns.str.strip()

        # Clean and convert Volume
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(str).str.replace(',', '', regex=False)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        # Clean and convert % Change
        if '% Change' in df.columns:
            df['% Change'] = df['% Change'].astype(str).str.replace('%', '', regex=False)
            df['% Change'] = pd.to_numeric(df['% Change'], errors='coerce') / 100

        # Convert prices to numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Optional: Handle Date if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            df.set_index('Date', inplace=True)

        # Drop rows where core prices are still missing
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)


        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Your file must include: {required_cols}")
            st.stop()

        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Returns'] = df['Close'].pct_change()
        df['Log_Volume'] = np.log(df['Volume'].clip(lower=1))

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD'] = macd
        df['MACD_Signal'] = signal

        df.dropna(inplace=True)
        st.success(f"✅ Cleaned data shape: {df.shape}")
        st.dataframe(df.tail())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned Data", csv, "cleaned_financial_data.csv")

        if df.shape[0] < 60:
            st.warning("⚠️ Not enough data for training. LSTM may perform poorly.")

        tab1, tab2 = st.tabs(["📈 LSTM Forecasting", "📉 GARCH Risk Forecast"])

        with tab1:
            st.subheader("📈 LSTM Price Forecast")
            try:
                features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
                X, y = create_sequences(df[features], target_col='Close')
                if len(X) == 0:
                    st.warning("⚠️ Not enough data for sequence modeling.")
                else:
                    split = int(len(X) * 0.8)
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                    model.fit(X_train, y_train, epochs=10, batch_size=16,
                              validation_data=(X_test, y_test),
                              callbacks=[EarlyStopping(patience=3)], verbose=0)
                    preds = model.predict(X_test).flatten()
                    st.line_chart({"Actual": y_test[:100], "Predicted": preds[:100]})
            except Exception as e:
                st.error(f"❌ LSTM Forecasting Error: {e}")

        with tab2:
            st.subheader("📉 GARCH Volatility Forecast")
            try:
                vol_forecast, var_1d = forecast_garch_var(df)
                st.metric("1-Day VaR (95%)", f"{var_1d:.2f}%")
                st.line_chart(vol_forecast.values)
                st.markdown("### 📖 Interpretation")
                st.info(f'''
                ✅ **Volatility Chart** shows predicted market turbulence over time.
                ✅ **Value at Risk (VaR)**: With 95% confidence, the expected 1-day loss will not exceed **{abs(var_1d):.2f}%**.
                Useful for risk managers, traders & investors.
                ''')
            except Exception as e:
                st.error(f"❌ GARCH Forecasting Error: {e}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👆 Upload a CSV with columns: Open, High, Low, Close, Volume")
