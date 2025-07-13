import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="📊 Test Financial App", layout="wide")
st.title("✅ Financial Dashboard is Working!")

st.markdown("Upload a CSV with at least the columns: `Open`, `High`, `Low`, `Close`, `Volume`")

uploaded_file = st.file_uploader("📤 Upload your file here", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Your file must include the following columns: {required_cols}")
        else:
            df['Returns'] = df['Close'].pct_change()
            df['Log_Volume'] = np.log(df['Volume'].replace(0, np.nan)).fillna(0)
            st.success(f"✅ Data Loaded: {df.shape[0]} rows")
            st.dataframe(df.tail())
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Upload a CSV file to get started.")

