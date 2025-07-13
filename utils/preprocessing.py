from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(df):
    # Select and scale numerical columns
    price_features = ['Open', 'High', 'Low', 'Close']
    volume_features = ['Volume']

    # Ensure correct types
    df[volume_features] = df[volume_features].apply(pd.to_numeric, errors='coerce')
    df['Log_Volume'] = np.log(df['Volume'].clip(lower=1))

    # Fill missing values
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df.dropna(inplace=True)

    # Scale prices
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    df[price_features] = minmax_scaler.fit_transform(df[price_features])

    return df, minmax_scaler, standard_scaler
