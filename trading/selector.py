"""Rank active strategies by estimated net, volatility-penalized return."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from .strategies import REASONS

FEATURES = ["momentum", "market", "relative", "slope", "trend", "strength",
            "vol", "atr", "volume", "rsi", "z", "breakout"]
HORIZON = 5
FAVORED = {
    "bullish": {"momentum", "trend", "breakout", "volatility"},
    "sideways": {"reversion", "pairs"},
    "bearish": {"pairs"}, "high volatility": {"volatility"}, "uncertain": set(),
}

def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

def training_rows(bars: pd.DataFrame, frame: pd.DataFrame, signals: pd.DataFrame,
                  friction: float) -> pd.DataFrame:
    # A candidate enters at the next open and exits at the first next-open
    # after its signal disappears, or at the five-session evaluation boundary.
    entry = bars.open.shift(-1)
    base_end = bars.index.to_series().shift(-(HORIZON + 1))
    samples = []
    for name, scale in signals.reindex(bars.index).items():
        exit_price = bars.open.shift(-(HORIZON + 1))
        label_end = base_end.copy()
        holding = pd.Series(HORIZON, index=bars.index)
        for offset in range(HORIZON - 1, 0, -1):
            exited = scale.shift(-offset).eq(0)
            exit_price = exit_price.mask(exited, bars.open.shift(-(offset + 1)))
            label_end = label_end.mask(exited, bars.index.to_series().shift(-(offset + 1)))
            holding = holding.mask(exited, offset)
        target = exit_price / entry - 1 - 2 * friction - .15 * frame.vol * np.sqrt(holding / 252)
        samples.append(prepare(frame.reindex(bars.index)).assign(
            strategy=name, target=target, label_end=label_end).loc[scale > 0])
    return pd.concat(samples).dropna(subset=["target", "label_end"])


def design(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[FEATURES].assign(**{name: (rows.strategy == name).astype(int)
                                   for name in REASONS})

class Selector:
    def __init__(self) -> None:
        self.model = None
        self.regime_model = None
        self.trained_at = None
        self.validation_mae = []

    def train(self, rows: pd.DataFrame, market_features: pd.DataFrame,
              market_regimes: pd.Series, calendar: pd.DatetimeIndex, cutoff: int) -> None:
        development = calendar[:cutoff]
        self.validation_mae = []
        for earlier, later in TimeSeriesSplit(n_splits=3, gap=HORIZON + 1).split(development):
            train = rows.loc[rows.index.isin(development[earlier]) & (rows.label_end < development[later[0]])]
            valid = rows.loc[rows.index.isin(development[later]) & (rows.label_end <= development[later[-1]])]
            if len(train) >= 30 and len(valid):
                model = self.fit(train)
                self.validation_mae.append(float(np.abs(model.predict(design(valid)) - valid.target).mean()))
        train = rows.loc[rows.index.isin(development[:-(HORIZON + 1)])]
        if len(train) < 30:
            raise ValueError("Insufficient active signals for training")
        self.model = self.fit(train)
        labels = market_regimes.reindex(development).dropna()
        if labels.nunique() > 1:
            self.regime_model = GradientBoostingClassifier(max_depth=2, random_state=42)
            self.regime_model.fit(prepare(market_features).loc[labels.index], labels)
        self.trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def classify(self, frame: pd.DataFrame) -> str | None:
        if self.regime_model is None:
            return None
        return str(self.regime_model.predict(prepare(frame))[0])

    @staticmethod
    def fit(rows: pd.DataFrame) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                         min_samples_leaf=12, random_state=42).fit(
                                             design(rows), rows.target)

    def rank(self, frame: pd.DataFrame, signals: pd.Series, label: str, method: str,
             previous: tuple[str, ...] = (), threshold: float = .001,
             count: int = 2) -> tuple[dict[str, float], dict[str, str]]:
        active = signals[signals > 0]
        explanations = {name: "entry conditions unmet" for name in signals.index}
        if method == "ml" and self.model is not None and len(active):
            rows = pd.concat([prepare(frame).assign(strategy=name) for name in active.index])
            scores = dict(zip(active.index, self.model.predict(design(rows))))
        elif method == "ml":
            scores = {}
        else:
            scores = {name: float(name in FAVORED[label]) if method == "rules" else 1.
                      for name in active.index}
        eligible = [name for name, score in scores.items()
                    if score >= threshold + (.0005 if method == "ml" and name not in previous else 0)]
        eligible.sort(key=lambda name: scores[name] + (.0005 if name in previous else 0), reverse=True)
        chosen = eligible if method == "equal" else eligible[:count]
        for name in active.index:
            score = scores.get(name)
            explanations[name] = (f"{'selected' if name in chosen else 'rejected'}; "
                                  f"score={score:.4f}; {REASONS[name]}" if score is not None
                                  else "model is unavailable")
        denominator = sum(scores[name] for name in chosen)
        return {name: float(active[name]) * scores[name] / denominator for name in chosen}, explanations
