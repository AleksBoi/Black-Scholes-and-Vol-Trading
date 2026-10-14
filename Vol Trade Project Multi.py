import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from scipy.stats import norm


def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C

def get_garch(log_returns):
    model = arch_model(log_returns, vol='Garch', p=1, q=1, rescale=False)
    res = model.fit(disp='off')
    return pd.Series(res.conditional_volatility.values * np.sqrt(252), index=log_returns.index)

def compute_prices(ticker, r_series, t):
    table = yf.download(tickers=ticker, interval='1d', period='5y', auto_adjust=True)
    data = table["Close"][ticker]
    log_returns = np.log(data / data.shift(1)).dropna()

    sig_g = get_garch(log_returns)
    sig_e = log_returns.ewm(alpha=0.06).std().dropna() * np.sqrt(252)
    sig_s = log_returns.rolling(30).std() * np.sqrt(252)

    df = pd.DataFrame({
        'S': data,
        'r': r_series,
        'sig_g': sig_g,
        'sig_e': sig_e,
        'sig_s': sig_s,
    }).dropna()

    prices_g = black_scholes_call(df['S'], df['S'], df['sig_g'], df['r'], t)
    prices_e = black_scholes_call(df['S'], df['S'], df['sig_e'], df['r'], t)
    prices_s = black_scholes_call(df['S'], df['S'], df['sig_s'], df['r'], t)

    return df.index, prices_g, prices_e, prices_s


tickers = ['AAPL', 'AMZN']
t = 1/12

irx = yf.download(tickers='^IRX', interval='1d', period='5y', auto_adjust=True)
r_series = irx["Close"]["^IRX"] / 100

colors = {'AAPL': ('tab:blue', 'tab:cyan', 'tab:purple'),
          'AMZN': ('tab:orange', 'tab:red', 'tab:green')}

plt.figure(figsize=(14, 6))

for ticker in tickers:
    idx, prices_g, prices_e, prices_s = compute_prices(ticker, r_series, t)
    c_g, c_e, c_s = colors[ticker]
    plt.plot(idx, prices_g, color=c_g, label=f'{ticker} GARCH')
    plt.plot(idx, prices_e, color=c_e, label=f'{ticker} EWMA')
    plt.plot(idx, prices_s, color=c_s, label=f'{ticker} Rolling 30d')

plt.title('ATM 1-Month Call Price (Black-Scholes) by Volatility Method')
plt.xlabel('Date')
plt.ylabel('Call Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()
