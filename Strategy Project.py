import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model

VOL_TARGET = 0.1
MAX_LEVERAGE = 1.0
INITIAL_CAPITAL = 100  # base = 100% of initial investment
UPM_Q = 1
LPM_N = 1  # n=2 squares losses: tiny in bull markets, large in crises → high ratio discrimination
UPM_LPM_WINDOW = 252
UPM_LPM_THRESHOLD = 1  # enter fully when UPM/LPM > this; natural breakeven at 1
TRAIN_MIN_DAYS = 252 * 3
RETRAIN_FREQ = 63  # quarterly
EPS = 1e-12
EWMA_ALPHA = 0.06
ROLLING_WINDOW = 30


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

df = pd.DataFrame({
    'S': data,
    'r': r_series,
    'sig_g': sig_g,
}).dropna()


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
        h = (df['r'] / 252).reindex(log_returns_aligned.index)
        ratio = upm_lpm_ratio(log_returns_aligned, h)
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

    print(ret_a)
    plt.figure(figsize=(14, 5))
    plt.plot(equity_a.index, equity_a, label='Rule A: Inv-Vol', linewidth=1.2)
    plt.plot(equity_b.index, equity_b, label='Rule B: Inv-Vol + Autocorr Overlay', linewidth=1.2)
    plt.plot(equity_c.index, equity_c, label=f'Rule C: UPM/LPM Screen (threshold={UPM_LPM_THRESHOLD})', linewidth=1.2)
    plt.plot(spy_equity.index, spy_equity, label='SPY buy-and-hold', linestyle='--', linewidth=1.0, alpha=0.7)
    plt.title('Rule A vs Rule B vs Rule C vs SPY')
    plt.xlabel('Date')
    plt.ylabel('% of Initial Investment')
    plt.legend()
    plt.tight_layout()

    diff_a = equity_a - spy_equity
    diff_b = equity_b - spy_equity
    diff_c = equity_c - spy_equity

    for name, diff in [('Rule A', diff_a), ('Rule B', diff_b), ('Rule C', diff_c)]:
        peak_date = diff.idxmax()
        trough_date = diff.idxmin()
        print(f"{name}: peak outperformance {diff.loc[peak_date]:+.1f} pts on {peak_date.date()}, "
              f"peak underperformance {diff.loc[trough_date]:+.1f} pts on {trough_date.date()}")

    plt.figure(figsize=(14, 5))
    plt.axhline(0, color='black', linewidth=0.8)
    plt.plot(diff_a.index, diff_a, label='Rule A - SPY', linewidth=1.2)
    plt.plot(diff_b.index, diff_b, label='Rule B - SPY', linewidth=1.2)
    plt.plot(diff_c.index, diff_c, label='Rule C - SPY', linewidth=1.2)
    plt.title('Strategy Outperformance vs SPY (Percentage Points)')
    plt.xlabel('Date')
    plt.ylabel('Difference (% of Initial Investment)')
    plt.legend()
    plt.tight_layout()
    plt.show()


strategy_testing()
