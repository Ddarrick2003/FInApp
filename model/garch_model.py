from arch import arch_model

def forecast_garch_var(df):
    returns = df['Returns'].dropna() * 100
    am = arch_model(returns, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    forecast = res.forecast(horizon=5)
    volatility = forecast.variance.iloc[-1] ** 0.5
    var_1d = -1.65 * volatility[0]
    return volatility, var_1d
