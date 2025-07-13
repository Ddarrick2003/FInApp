from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, target_col, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i:i + window].values)
        y.append(data.iloc[i + window][target_col])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
