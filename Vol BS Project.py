import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from scipy.stats import norm
import tkinter as tk
from tkinter import ttk

OPTION_MATURITY = 1 / 12  # 1-month ATM call


def get_params():
    root = tk.Tk()
    root.title("Parameters")
    root.resizable(False, False)

    tk.Label(root, text="EWMA Alpha (0–1):").grid(row=0, column=0, padx=12, pady=8, sticky="w")
    alpha_var = tk.StringVar(value="0.06")
    tk.Entry(root, textvariable=alpha_var, width=10).grid(row=0, column=1, padx=12, pady=8)

    tk.Label(root, text="Rolling Window (days):").grid(row=1, column=0, padx=12, pady=8, sticky="w")
    window_var = tk.StringVar(value="30")
    tk.Entry(root, textvariable=window_var, width=10).grid(row=1, column=1, padx=12, pady=8)

    result = {}

    def submit():
        result["alpha"] = float(alpha_var.get())
        result["window"] = int(window_var.get())
        root.destroy()

    ttk.Button(root, text="Generate", command=submit).grid(row=2, column=0, columnspan=2, pady=12)
    root.mainloop()
    return result["alpha"], result["window"]


EWMA_ALPHA, ROLLING_WINDOW = get_params()


def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)


def get_garch(log_returns):
    r = log_returns.dropna() * 100
    model = arch_model(r, vol='Garch', p=1, q=1, rescale=False)
    res = model.fit(disp='off')
    cv = res.conditional_volatility / 100
    return pd.Series(cv.values * np.sqrt(252), index=r.dropna().index)


spy = yf.download(tickers='SPY', start='2000-01-01', interval='1d', auto_adjust=True)
data = spy['Close'].squeeze().tz_localize('UTC')

irx = yf.download(tickers='^IRX', start='2000-01-01', interval='1d', auto_adjust=True)
r_series = irx['Close'].squeeze().tz_localize('UTC') / 100

log_returns = np.log(data / data.shift(1)).dropna()

sig_g = get_garch(log_returns)
sig_e = log_returns.ewm(alpha=EWMA_ALPHA).std() * np.sqrt(252)
sig_s = log_returns.rolling(ROLLING_WINDOW).std() * np.sqrt(252)

df = pd.DataFrame({
    'S': data,
    'r': r_series,
    'sig_g': sig_g,
    'sig_e': sig_e,
    'sig_s': sig_s,
}).dropna()

prices_g = black_scholes_call(df['S'], df['S'], df['sig_g'], df['r'], OPTION_MATURITY)
prices_e = black_scholes_call(df['S'], df['S'], df['sig_e'], df['r'], OPTION_MATURITY)
prices_s = black_scholes_call(df['S'], df['S'], df['sig_s'], df['r'], OPTION_MATURITY)


def plot_volatilities():
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df.index, df['sig_g'] * 100, label='GARCH(1,1)', linewidth=1.0)
    ax1.plot(df.index, df['sig_e'] * 100, label=f'EWMA (α={EWMA_ALPHA})', linewidth=0.8, alpha=0.85)
    ax1.plot(df.index, df['sig_s'] * 100, label=f'Rolling {ROLLING_WINDOW}d', linewidth=0.8, alpha=0.85)
    ax1.set_title('SPY Annualised Volatility — Three Estimators')
    ax1.set_ylabel('Annualised Volatility (%)')
    ax1.legend()

    ax2.plot(df.index, prices_g, label='BS Price (GARCH)', linewidth=1.0)
    ax2.plot(df.index, prices_e, label=f'BS Price (EWMA α={EWMA_ALPHA})', linewidth=0.8, alpha=0.85)
    ax2.plot(df.index, prices_s, label=f'BS Price (Rolling {ROLLING_WINDOW}d)', linewidth=0.8, alpha=0.85)
    ax2.set_title('ATM Call Price (Black-Scholes) — Three Volatility Estimators')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Option Price (USD)')

    spy_scale = round(df['S'].mean() / prices_g.mean(), -1)
    ax2.plot(df.index, df['S'] / spy_scale, label=f'SPY / {spy_scale:.0f}', color='black', linewidth=0.8, alpha=0.4, linestyle='--')
    ax2.legend()

    plt.tight_layout()
    plt.show()


plot_volatilities()
