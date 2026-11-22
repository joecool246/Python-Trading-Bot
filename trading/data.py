"""Validated OHLCV and causal daily indicators."""
import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume"]

def load(path: str) -> dict[str, pd.DataFrame]:
    rows = pd.read_csv(path)
    rows["symbol"] = rows["symbol"].str.strip().str.upper()
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    if rows.empty or rows["symbol"].isna().any() or (rows["symbol"] == "").any() or rows["date"].isna().any():
        raise ValueError("Missing symbols or dates")
    if rows.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate bars")
    values = rows[FIELDS].astype(float)
    valid = (np.isfinite(values).all(axis=1) & (values.iloc[:, :4] > 0).all(axis=1)
             & (values.volume >= 0)
             & (values.high >= values[["open", "close", "low"]].max(axis=1))
             & (values.low <= values[["open", "close"]].min(axis=1)))
    if not valid.all():
        raise ValueError("Invalid OHLCV")
    rows[FIELDS] = values
    return {symbol: group.set_index("date")[FIELDS].sort_index()
            for symbol, group in rows.groupby("symbol")}

def synthetic() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(128)
    dates = pd.bdate_range("2020-01-01", periods=780, tz="UTC")
    common = rng.normal(0.0002, .01, len(dates)) + .001 * np.sin(np.arange(len(dates)) / 25)
    result = {}
    for index, symbol in enumerate(("SPY", "DEMO", "SAMPLE")):
        close = 100 * np.exp(np.cumsum(common + index * rng.normal(0, .004, len(dates))))
        opening = close * np.exp(rng.normal(0, .003, len(dates)))
        result[symbol] = pd.DataFrame(dict(open=opening, close=close,
            high=np.maximum(opening, close) * 1.005, low=np.minimum(opening, close) * .995,
            volume=rng.integers(10000, 50000, len(dates))), index=dates)
    return result

def features(bars: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    close = bars.close
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    true_range = pd.concat([bars.high - bars.low, (bars.high - close.shift()).abs(),
                            (bars.low - close.shift()).abs()], axis=1).max(axis=1)
    average = close.rolling(20).mean()
    returns = close.pct_change(fill_method=None)
    market = benchmark.pct_change(20, fill_method=None).reindex(close.index)
    return pd.DataFrame(dict(momentum=close.pct_change(20, fill_method=None),
        market=market, relative=close.pct_change(20, fill_method=None) - market,
        slope=average.pct_change(5, fill_method=None), trend=close / average - 1,
        strength=delta.rolling(20).sum() / delta.abs().rolling(20).sum().replace(0, np.nan),
        vol=returns.rolling(20).std() * np.sqrt(252), atr=true_range.rolling(14).mean() / close,
        volume=bars.volume / bars.volume.rolling(20).mean().replace(0, np.nan),
        rsi=(100 * gain / (gain + loss)).mask((gain + loss) == 0, 50),
        z=(close - average) / close.rolling(20).std().replace(0, np.nan),
        breakout=close / bars.high.shift().rolling(20).max() - 1))
