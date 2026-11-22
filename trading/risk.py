"""Independent sizing and execution gates."""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class Limits:
    position: float = .15
    exposure: float = .65
    sector: float = .35
    daily_loss: float = .025
    drawdown: float = .12
    trade_risk: float = .006
    max_orders: int = 8
    max_age_hours: float = 30
    kill: bool = False
    max_spread: float = .02

    def __post_init__(self) -> None:
        if any(not 0 < value <= 1 for value in (self.position, self.exposure,
               self.sector, self.daily_loss, self.drawdown, self.trade_risk, self.max_spread)):
            raise ValueError("Risk fractions must be between zero and one")
        if self.max_orders < 1 or not isfinite(self.max_age_hours) or self.max_age_hours <= 0:
            raise ValueError("Invalid order frequency or data age limit")

@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    exposure: float
    peak: float
    session_start: float
    buying_power: float

def approve(limits: Limits, account: Account, held: int, requested: int, price: float,
            atr: float, *, age_hours: float = 0, orders: int = 0, pending: bool = False,
            market_open: bool = True, healthy: bool = True,
            sector_value: float | None = None) -> tuple[int, str]:
    inputs = (account.equity, account.cash, account.exposure, account.peak,
              account.session_start, account.buying_power, price, atr, age_hours)
    if (limits.kill or not healthy or not all(map(isfinite, inputs))
            or account.equity <= 0 or account.peak <= 0 or account.session_start <= 0
            or account.exposure < 0 or price <= 0 or atr < 0 or held < 0
            or (sector_value is not None and (not isfinite(sector_value) or sector_value < 0))):
        return 0, "kill switch or invalid account/data"
    if pending or not market_open or not 0 <= age_hours <= limits.max_age_hours or orders >= limits.max_orders:
        return 0, "open order, outside session, stale data or frequency limit"
    if requested < 0:
        return max(requested, -held), "reduce existing exposure"
    equity = account.equity
    if equity <= account.session_start * (1 - limits.daily_loss) or equity <= account.peak * (1 - limits.drawdown):
        return 0, "daily loss or drawdown limit"
    capacity = min(account.cash, account.buying_power, limits.position * equity - held * price,
                   limits.exposure * equity - account.exposure,
                   limits.trade_risk * equity / max(2 * atr, .01))
    if sector_value is not None:
        capacity = min(capacity, limits.sector * equity - sector_value)
    quantity = min(requested, max(0, int(capacity / price)))
    return quantity, "approved" if quantity else "position, exposure, buying power or risk limit"
