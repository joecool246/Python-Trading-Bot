"""Long-only strategy signals and interpretable market regimes."""
import numpy as np
import pandas as pd

def regime(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select(
        [frame[["vol", "slope", "strength"]].isna().any(axis=1), frame.vol > .32,
         (frame.slope > .003) & (frame.strength > .2),
         (frame.slope < -.003) & (frame.strength < -.2)],
        ["uncertain", "high volatility", "bullish", "bearish"], default="sideways"),
        index=frame.index)

def candidates(frame: pd.DataFrame, pair_z: pd.Series, config: dict) -> pd.DataFrame:
    """Each positive cell is a desired exposure multiplier for that strategy."""
    return pd.DataFrame({
        "momentum": ((frame.momentum > config.get("momentum", .02))
                     & (frame.relative > 0) & (frame.volume > .8)).astype(float),
        "reversion": ((frame.rsi < config.get("rsi", 35))
                      & (frame.z < config.get("band", -1.5))).astype(float),
        "trend": ((frame.trend > 0) & (frame.slope > .003)
                  & (frame.strength > config.get("strength", .2))).astype(float),
        "breakout": ((frame.breakout > 0) & (frame.volume > config.get("breakout_volume", 1.1))).astype(float),
        "pairs": (pair_z < config.get("pair_entry", -2)).astype(float),
        "volatility": ((frame.trend > 0) & (frame.vol < config.get("vol_cap", .35)))
                       * (.015 / frame.atr).clip(0, 1),
    }, index=frame.index).fillna(0)

REASONS = {
    "momentum": "positive momentum and relative strength with volume",
    "reversion": "low RSI and close below the rolling price band",
    "trend": "positive moving-average slope and directional strength",
    "breakout": "close above the previous range with volume confirmation",
    "pairs": "discount to a screened cointegrating relationship",
    "volatility": "positive trend with ATR-adjusted exposure",
}

def pair_spread(close: pd.Series, reference: pd.Series, window: int = 90) -> pd.Series:
    """Fit on strictly prior bars and refresh the screen every 20 observations."""
    from statsmodels.tsa.stattools import coint
    joined = pd.concat([np.log(close), np.log(reference.reindex(close.index))], axis=1)
    output = pd.Series(np.nan, index=close.index)
    fit = None
    for step in range(window, len(joined)):
        if (step - window) % 20 == 0:
            fit = None
            history = joined.iloc[step-window:step].dropna()
            if len(history) == window and history.corr().iloc[0, 1] > .6:
                y, x = history.to_numpy().T
                if coint(y, x)[1] < .05:
                    beta, intercept = np.polyfit(x, y, 1)
                    spread = y - beta * x - intercept
                    if beta > 0 and spread.std() > 1e-6:
                        fit = beta, intercept, spread.std()
        if fit is not None:
            beta, intercept, scale = fit
            y, x = joined.iloc[step]
            output.iloc[step] = (y - beta * x - intercept) / scale
    return output
