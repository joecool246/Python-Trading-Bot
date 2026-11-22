"""Performance reporting for simulated fills and marked equity."""
import numpy as np
import pandas as pd

def daily_returns(curve: pd.DataFrame, initial: float) -> pd.Series:
    equity = curve.equity
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / initial - 1
    return returns

def metrics(curve: pd.DataFrame, fills: pd.DataFrame, initial: float) -> dict:
    equity = curve.equity
    returns = daily_returns(curve, initial)
    total = equity.iloc[-1] / initial
    volatility = returns.std()
    downside = np.sqrt(np.mean(returns.clip(upper=0) ** 2))
    pnl = fills.pnl.dropna() if "pnl" in fills else pd.Series(dtype=float)
    gains, losses = pnl.clip(lower=0).sum(), -pnl.clip(upper=0).sum()
    sales = fills.loc[fills.qty < 0] if "holding_sessions" in fills else None
    holding = (np.average(sales.holding_sessions, weights=-sales.qty) if sales is not None
               and len(sales) else None)
    by_symbol = {symbol: float(items.pnl.dropna().sum())
                 for symbol, items in fills.groupby("symbol")} if len(fills) else {}
    allocation_by_regime = {}
    if "allocation" in curve:
        weights = pd.DataFrame(curve.allocation.tolist(), index=curve.index).fillna(0)
        allocation_by_regime = weights.groupby(curve.regime).mean().to_dict(orient="index")
    return dict(total_return=float(total - 1), annual_return=float(total ** (252 / len(curve)) - 1),
        sharpe=float(np.sqrt(252) * returns.mean() / volatility) if volatility > 0 else None,
        sortino=float(np.sqrt(252) * returns.mean() / downside) if downside > 0 else None,
        max_drawdown=float((equity / equity.cummax().clip(lower=initial) - 1).min()),
        volatility=float(volatility * np.sqrt(252)), win_rate=float((pnl > 0).mean()) if len(pnl) else None,
        profit_factor=float(gains / losses) if losses else None,
        turnover=float((fills.qty.abs() * fills.price).sum() / equity.mean()) if len(fills) else 0,
        costs=float(fills.cost.sum()) if len(fills) else 0, fills=len(fills),
        average_sale_holding_sessions=float(holding) if holding is not None else None,
        average_exposure=float((curve.exposure / equity).mean()),
        realized_pnl_by_symbol=by_symbol, allocation_by_regime=allocation_by_regime,
        by_regime={label: {"sessions": len(subset),
                           "compounded_return": float((1 + subset).prod() - 1)}
                   for label, subset in returns.groupby(curve.regime)})

def paired_edge(candidate: pd.Series, baseline: pd.Series, hurdle: float = .01,
                seed: int = 42, draws: int = 2000) -> dict:
    """Paired circular block bootstrap of annualized arithmetic active return."""
    if not candidate.index.equals(baseline.index):
        raise ValueError("Comparisons require identical sessions")
    active = (candidate - baseline).to_numpy()
    n = len(active)
    block = min(10, n)
    center = float(252 * active.mean())
    result = {"annualized_active_return": center, "materiality_hurdle": hurdle,
              "sessions": n, "block_sessions": block, "interval_95": None,
              "clears_hurdle": None}
    if n >= 60:
        rng = np.random.default_rng(seed)
        starts = rng.integers(n, size=(draws, (n + block - 1) // block))
        positions = (starts[..., None] + np.arange(block)) % n
        estimates = 252 * active[positions.reshape(draws, -1)[:, :n]].mean(axis=1)
        low, high = np.quantile(estimates, [.025, .975])
        result.update(interval_95=[float(low), float(high)], clears_hurdle=bool(low > hurdle))
    return result

def attribution(fills: pd.DataFrame) -> dict[str, float]:
    realized = {}
    for fill in fills.itertuples():
        if pd.notna(fill.pnl):
            total = sum(fill.allocation.values())
            for name, value in fill.allocation.items():
                realized[name] = realized.get(name, 0.) + fill.pnl * value / total
    return realized

def paired_sharpe(candidate: pd.Series, baseline: pd.Series, seed: int = 42,
                  draws: int = 2000, block_sessions: int = 10) -> dict:
    """Bootstrap the difference between annualized zero-rate Sharpe ratios."""
    if not candidate.index.equals(baseline.index):
        raise ValueError("Comparisons require identical sessions")
    values = np.stack([candidate.to_numpy(), baseline.to_numpy()], axis=-1)
    def difference(rows):
        deviation = rows.std(axis=-2, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.sqrt(252) * rows.mean(axis=-2) / deviation
        return sharpe[..., 0] - sharpe[..., 1]
    point = difference(values)
    result = {"difference": float(point) if np.isfinite(point) else None, "interval_95": None}
    if len(values) >= 60:
        rng = np.random.default_rng(seed)
        starts = rng.integers(len(values), size=(draws, (len(values) + block_sessions - 1) // block_sessions))
        positions = (starts[..., None] + np.arange(block_sessions)) % len(values)
        estimates = difference(values[positions.reshape(draws, -1)[:, :len(values)]])
        valid = estimates[np.isfinite(estimates)]
        if len(valid) >= .95 * draws:
            result["interval_95"] = np.quantile(valid, [.025, .975]).tolist()
    return result
