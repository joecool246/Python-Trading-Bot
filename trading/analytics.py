"""Performance reporting for simulated fills and marked equity."""
import numpy as np
import pandas as pd

def metrics(curve: pd.DataFrame, fills: pd.DataFrame, initial: float) -> dict:
    equity = curve.equity
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / initial - 1
    total = equity.iloc[-1] / initial
    volatility = returns.std()
    downside = np.sqrt(np.mean(returns.clip(upper=0) ** 2))
    pnl = fills.pnl.dropna() if "pnl" in fills else pd.Series(dtype=float)
    gains, losses = pnl.clip(lower=0).sum(), -pnl.clip(upper=0).sum()
    return dict(total_return=float(total - 1), annual_return=float(total ** (252 / len(curve)) - 1),
        sharpe=float(np.sqrt(252) * returns.mean() / volatility) if volatility > 0 else None,
        sortino=float(np.sqrt(252) * returns.mean() / downside) if downside > 0 else None,
        max_drawdown=float((equity / equity.cummax().clip(lower=initial) - 1).min()),
        volatility=float(volatility * np.sqrt(252)), win_rate=float((pnl > 0).mean()) if len(pnl) else None,
        profit_factor=float(gains / losses) if losses else None,
        turnover=float((fills.qty.abs() * fills.price).sum() / equity.mean()) if len(fills) else 0,
        costs=float(fills.cost.sum()) if len(fills) else 0,
        by_regime={label: float((1 + subset).prod() - 1)
                   for label, subset in returns.groupby(curve.regime)})

def attribution(fills: pd.DataFrame) -> dict[str, float]:
    realized = {}
    for fill in fills.itertuples():
        if pd.notna(fill.pnl):
            total = sum(fill.allocation.values())
            for name, value in fill.allocation.items():
                realized[name] = realized.get(name, 0.) + fill.pnl * value / total
    return realized
