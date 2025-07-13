from arch import arch_model
import pandas as pd

def forecast_garch_var(df, p=1, q=1):
    returns = df['Returns'].dropna() * 100  # Scale to percentage

    am = arch_model(returns, vol='Garch', p=p, q=q)
    res = am.fit(disp='off')

    forecast = res.forecast(horizon=5)
    vol_forecast = forecast.variance.values[-1, :] ** 0.5
    var_1d = res.conditional_volatility[-1] * 1.65  # 95% VaR

    forecast_df = pd.Series(vol_forecast, name="Volatility Forecast")

    return forecast_df, var_1d
