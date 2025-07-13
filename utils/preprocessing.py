from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

def preprocess_data(df):
    df = df.copy()
    df.dropna(inplace=True)
    df = df[df['Volume'] > 0]
    if df.empty:
        raise ValueError("After cleaning, dataframe is empty. Check your CSV content.")

    price_features = ['Open', 'High', 'Low', 'Close']
    minmax_scaler = MinMaxScaler()
    df[price_features] = minmax_scaler.fit_transform(df[price_features])

    standard_scaler = StandardScaler()
    df[['Log_Volume', 'RSI', 'MACD', 'Returns']] = standard_scaler.fit_transform(
        df[['Log_Volume', 'RSI', 'MACD', 'Returns']]
    )
    return df, minmax_scaler, standard_scaler
