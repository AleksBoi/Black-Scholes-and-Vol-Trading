import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOL_TARGET = 0.10
MAX_LEVERAGE = 1.0
INITIAL_CAPITAL = 1000
UPM_LPM_WINDOW = 252
TRAIN_MIN_DAYS = 252 * 3
RETRAIN_FREQ = 63

QN_PROFILES = [
    (0.25, 1, 'Risk Averse (q=0.25, n=1)'),
    (0.44, 1, 'Prospect Theory (q=0.44, n=1)'),
    (1.0,  1, 'Risk Neutral (q=1, n=1)'),
    (2.0,  1, 'Risk Seeking (q=2, n=1)'),
]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
spy = yf.download(tickers='SPY', start='2000-01-01', interval='1d', auto_adjust=True)
data = spy['Close'].squeeze().tz_localize('UTC')

irx = yf.download(tickers='^IRX', start='2000-01-01', interval='1d', auto_adjust=True)
r_series = irx['Close'].squeeze().tz_localize('UTC') / 100  # annualised rf

log_returns = np.log(data / data.shift(1)).dropna()

# GARCH(1,1) annualised conditional vol (rescale by hand, return to decimal)
model = arch_model(log_returns * 100, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
res = model.fit(disp='off')
sig_g = pd.Series(
    res.conditional_volatility.values / 100 * np.sqrt(252),
    index=log_returns.index,
)

df = pd.DataFrame({'S': data, 'r': r_series, 'sig_g': sig_g}).dropna()
log_returns_aligned = np.log(df['S'] / df['S'].shift(1))
daily_rf = df['r'] / 252  # daily risk-free return (decimal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_position(position, asset_ret, rf):
    """
    Combine a position series (fraction of capital in the risky asset) with
    asset returns and risk-free cash on the unallocated portion.

    position is assumed to already be lagged (i.e. decided at t-1, applied at t).
    """
    position = position.clip(upper=MAX_LEVERAGE).fillna(0.0)
    cash_weight = (1.0 - position).clip(lower=0.0)  # no negative cash on leverage
    return (position * asset_ret + cash_weight * rf).dropna()


# ---------------------------------------------------------------------------
# Rule A: simple inverse-vol target
# ---------------------------------------------------------------------------
def simple_inv_vol():
    position = (VOL_TARGET / df['sig_g']).shift(1)
    return apply_position(position, log_returns_aligned, daily_rf)


# ---------------------------------------------------------------------------
# Rule C: UPM/LPM screened inv-vol
# ---------------------------------------------------------------------------
def upm_lpm_ratio(returns, h, q, n):
    excess_up = (returns - h).clip(lower=0)
    excess_dn = (h - returns).clip(lower=0)
    upm = (excess_up ** q).rolling(UPM_LPM_WINDOW).mean()
    lpm = (excess_dn ** n).rolling(UPM_LPM_WINDOW).mean()
    return upm / lpm.replace(0, np.nan)


def upm_lpm_screened(q, n):
    h_t = daily_rf
    # Lag the signal: decision at t uses information up to t-1.
    ratio = upm_lpm_ratio(log_returns_aligned, h_t, q, n).shift(1)

    position_base = (VOL_TARGET / df['sig_g']).shift(1)
    positions = pd.Series(np.nan, index=df.index)
    all_dates = df.index
    train_end = TRAIN_MIN_DAYS

    while train_end < len(all_dates):
        test_start = train_end
        test_end = min(train_end + RETRAIN_FREQ, len(all_dates))

        train_ratio = ratio.iloc[:train_end].dropna()
        train_returns = log_returns_aligned.iloc[:train_end]
        train_pos_base = position_base.iloc[:train_end]
        train_rf = daily_rf.iloc[:train_end]

        # Always evaluate "no screen" (threshold = -inf) so the optimiser can
        # decline to filter at all if filtering hurts.
        candidate_threshs = [-np.inf] + list(
            np.percentile(train_ratio, [25, 50, 75])
        )

        best_thresh, best_sharpe = -np.inf, -np.inf
        for thresh in candidate_threshs:
            pos = train_pos_base.copy()
            pos[ratio.iloc[:train_end] < thresh] = 0.0
            full_ret = apply_position(pos, train_returns, train_rf)
            if len(full_ret) < 30 or full_ret.std() == 0:
                continue
            sharpe = full_ret.mean() / full_ret.std() * np.sqrt(252)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_thresh = thresh

        test_idx = all_dates[test_start:test_end]
        test_pos = position_base.loc[test_idx].copy()
        test_pos[ratio.loc[test_idx] < best_thresh] = 0.0
        positions.loc[test_idx] = test_pos

        train_end = test_end

    return apply_position(positions, log_returns_aligned, daily_rf)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
ret_a = simple_inv_vol()
results = {'Rule A: Inv-Vol': ret_a}
for q, n, label in QN_PROFILES:
    print(f'Running Rule C: {label}...')
    results[f'Rule C — {label}'] = upm_lpm_screened(q, n)

common_idx = ret_a.index
for ret in results.values():
    common_idx = common_idx.intersection(ret.index)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(
    common_idx,
    (1 + ret_a.loc[common_idx]).cumprod() * INITIAL_CAPITAL,
    label='Rule A: Inv-Vol', linewidth=2.0, color='black',
)

colors = ['steelblue', 'darkorange', 'green', 'crimson']
for (q, n, label), color in zip(QN_PROFILES, colors):
    ret = results[f'Rule C — {label}']
    equity = (1 + ret.loc[common_idx]).cumprod() * INITIAL_CAPITAL
    plt.plot(common_idx, equity, label=f'Rule C — {label}',
             linewidth=1.2, color=color)

spy_equity = (1 + log_returns_aligned.loc[common_idx]).cumprod() * INITIAL_CAPITAL
plt.plot(common_idx, spy_equity, label='SPY buy-and-hold',
         linestyle='--', linewidth=1.0, color='grey', alpha=0.8)

plt.title('Rule A vs Rule C across (q, n) profiles')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def stats(ret):
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + ret).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    return ann_ret, ann_vol, sharpe, dd


print(f"\n{'Strategy':<45} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>8}")
print('-' * 81)
for name, ret in results.items():
    ar, av, sh, dd = stats(ret.loc[common_idx])
    print(f'{name:<45} {ar:>8.2%} {av:>8.2%} {sh:>8.2f} {dd:>8.2%}')

spy_ret_aligned = log_returns_aligned.loc[common_idx]
ar, av, sh, dd = stats(spy_ret_aligned)
print(f"{'SPY buy-and-hold':<45} {ar:>8.2%} {av:>8.2%} {sh:>8.2f} {dd:>8.2%}")