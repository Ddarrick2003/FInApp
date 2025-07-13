from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

def preprocess_data(df):
    price_features = ['Open', 'High', 'Low', 'Close']
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Log_Volume'] = np.log(df['Volume'].clip(lower=1))
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df.dropna(inplace=True)
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    df[price_features] = minmax_scaler.fit_transform(df[price_features])
    return df, minmax_scaler, standard_scaler
