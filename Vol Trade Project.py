import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from scipy.stats import norm

EWMA_ALPHA = 0.06
ROLLING_WINDOW = 30
OPTION_MATURITY = 1 / 12  # 1-month ATM call
VOL_TARGET = 0.1
MAX_LEVERAGE = 1.0
INITIAL_CAPITAL = 1000
UPM_Q = 1
LPM_N = 1  # n=2 squares losses: tiny in bull markets, large in crises → high ratio discrimination
UPM_LPM_WINDOW = 252
UPM_LPM_THRESHOLD = 1  # enter fully when UPM/LPM > this; natural breakeven at 1
TRAIN_MIN_DAYS = 252 * 3
RETRAIN_FREQ = 63  # quarterly
EPS = 1e-12


def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)


def get_garch(log_returns):
    r = log_returns.dropna() * 100  # percent returns
    model = arch_model(r, vol='Garch', p=1, q=1, rescale=False)
    res = model.fit(disp='off')
    cv = res.conditional_volatility / 100  # back to decimal
    return pd.Series(cv.values * np.sqrt(252), index=r.dropna().index)


spy = yf.download(tickers='INTL', start='2000-01-01', interval='1d', auto_adjust=True)
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
    #ax1.plot(df.index, df['r'] * 100, label='Risk-Free Rate (^IRX)', linewidth=0.8, alpha=0.9, linestyle='--', color='gray')
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


def strategy_testing():
    log_returns_aligned = np.log(df['S'] / df['S'].shift(1))

    def simple_inv_vol():
        position = (VOL_TARGET / df['sig_g']).shift(1).clip(upper=MAX_LEVERAGE)
        return (position * log_returns_aligned).dropna()

    def autocorrection():
        rho = log_returns.rolling(60).corr(log_returns.shift(1))
        position = ((VOL_TARGET / df['sig_g']).shift(1) * (1 - rho.abs()).shift(1)).clip(upper=MAX_LEVERAGE)
        return (position * log_returns_aligned).dropna()

    def upm_lpm_ratio(returns, h, q=UPM_Q, n=LPM_N):
        r = returns * 100
        h_scaled = h * 100
        excess_up = (r - h_scaled).clip(lower=0)
        excess_dn = (h_scaled - r).clip(lower=0)
        upm = (excess_up ** q).rolling(UPM_LPM_WINDOW).mean()
        lpm = (excess_dn ** n).rolling(UPM_LPM_WINDOW).mean()
        return upm / lpm.replace(0, EPS)

    
    def upm_lpm_screened():
        # Daily risk-free rate as the per-day target h
        h = (df['r'] / 252).reindex(log_returns_aligned.index)
        ratio = upm_lpm_ratio(log_returns_aligned, h)
        # Binary: fully enter at baseline inv-vol sizing when ratio > threshold, else flat
        signal = (ratio.shift(1) > UPM_LPM_THRESHOLD).astype(float)
        position = ((VOL_TARGET / df['sig_g']).shift(1) * signal).clip(upper=MAX_LEVERAGE)
        return (position * log_returns_aligned).dropna()

    ret_a = simple_inv_vol()
    ret_b = autocorrection()
    ret_c = upm_lpm_screened()

    h = (df['r'] / 252).reindex(log_returns_aligned.index)
    ratio = upm_lpm_ratio(log_returns_aligned, h)
    signal = (ratio.shift(1) > UPM_LPM_THRESHOLD).dropna()
    print(f"Rule C invested {signal.mean():.1%} of days ({signal.sum():.0f} / {len(signal)} trading days)")

    common_idx = ret_a.index.intersection(ret_b.index).intersection(ret_c.index)
    equity_a = np.exp(ret_a.loc[common_idx].cumsum()) * INITIAL_CAPITAL
    equity_b = np.exp(ret_b.loc[common_idx].cumsum()) * INITIAL_CAPITAL
    equity_c = np.exp(ret_c.loc[common_idx].cumsum()) * INITIAL_CAPITAL
    spy_equity = np.exp(log_returns_aligned.loc[common_idx].cumsum()) * INITIAL_CAPITAL

    plt.figure(figsize=(14, 5))
    plt.plot(equity_a.index, equity_a, label='Rule A: Inv-Vol', linewidth=1.2)
    plt.plot(equity_b.index, equity_b, label='Rule B: Inv-Vol + Autocorr Overlay', linewidth=1.2)
    plt.plot(equity_c.index, equity_c, label=f'Rule C: UPM/LPM Screen (threshold={UPM_LPM_THRESHOLD})', linewidth=1.2)
    plt.plot(spy_equity.index, spy_equity, label='SPY buy-and-hold', linestyle='--', linewidth=1.0, alpha=0.7)
    plt.title('Rule A vs Rule B vs Rule C vs SPY')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_volatilities()
#strategy_testing()
