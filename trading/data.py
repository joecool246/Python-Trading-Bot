"""Validated OHLCV and causal daily indicators."""
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume"]

def load(path: str) -> dict[str, pd.DataFrame]:
    return validate_rows(pd.read_csv(path, dtype={"symbol": "string"}))

def symbols_from(config: dict, default=()) -> list[str]:
    symbols = config.get("symbols", list(default))
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("Configure a nonempty symbols list")
    normalized = []
    for symbol in symbols:
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]*", symbol.strip().upper()):
            raise ValueError(f"Invalid stock symbol: {symbol!r}")
        normalized.append(symbol.strip().upper())
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate stock symbols")
    return normalized

def validate_rows(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = ["symbol", "date", *FIELDS]
    if missing := set(required) - set(rows.columns):
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    rows = rows[required].copy()
    rows["symbol"] = rows["symbol"].astype("string").str.strip().str.upper()
    rows["date"] = pd.to_datetime(rows["date"], format="mixed", utc=True).dt.normalize()
    if rows.empty or rows["symbol"].isna().any() or (rows["symbol"] == "").any() or rows["date"].isna().any():
        raise ValueError("Missing symbols or dates")
    if rows.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate bars")
    values = rows[FIELDS].astype(float)
    upper = values[["open", "close", "low"]].max(axis=1)
    lower = values[["open", "close"]].min(axis=1)
    # The same corporate-action adjustment can round neighboring prices differently.
    valid = (np.isfinite(values).all(axis=1) & (values.iloc[:, :4] > 0).all(axis=1)
             & (values.volume >= 0)
             & ((values.high >= upper) | np.isclose(values.high, upper, rtol=1e-12, atol=0))
             & ((values.low <= lower) | np.isclose(values.low, lower, rtol=1e-12, atol=0)))
    if not valid.all():
        raise ValueError("Invalid OHLCV")
    values["high"] = values.high.clip(lower=upper)
    values["low"] = values.low.clip(upper=lower)
    rows[FIELDS] = values
    return {symbol: group.set_index("date")[FIELDS].sort_index()
            for symbol, group in rows.groupby("symbol")}

def export_bars(adapter, symbols: list[str], start, end, path: str, *,
                feed: str = "iex", adjustment: str = "all") -> None:
    frame = adapter.bars(symbols, start, end, feed=feed, adjustment=adjustment)
    if frame.empty:
        raise ValueError("Alpaca returned no daily bars")
    rows = frame.reset_index().rename(columns={"timestamp": "date"})
    timestamps = pd.to_datetime(rows["date"], utc=True)
    # The endpoint's end is inclusive. Our end is an exclusive session boundary.
    rows = rows.loc[(timestamps >= start) & (timestamps < end)]
    validated = validate_rows(rows)
    if set(validated) != set(symbols):
        raise ValueError(f"Unexpected coverage. Missing: {sorted(set(symbols) - set(validated))}; "
                         f"unexpected: {sorted(set(validated) - set(symbols))}")
    daily = pd.concat(validated, names=["symbol"]).reset_index()
    output = Path(path)
    with NamedTemporaryFile(dir=output.parent, delete=False, suffix=".csv") as temporary:
        staged = Path(temporary.name)
    try:
        daily.to_csv(staged, index=False, date_format="%Y-%m-%d")
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)

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
