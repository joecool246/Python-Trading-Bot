"""Alpaca paper adapter and a durable, fail-closed order journal."""
import hashlib
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from typing import Protocol
from .risk import Account, Limits, approve

class Broker(Protocol):
    def snapshot(self) -> tuple: ...
    def submit(self, symbol: str, qty: int, price: float, order_id: str): ...
    def lookup(self, order_id: str): ...

class AlpacaPaper:
    def __init__(self, key: str, secret: str) -> None:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        self.client = TradingClient(key, secret, paper=True)
        self.data = StockHistoricalDataClient(key, secret)

    def snapshot(self) -> tuple:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        account = self.client.get_account()
        positions = self.client.get_all_positions()
        orders = self.client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        clock = self.client.get_clock()
        if account.trading_blocked or any(p.side.value != "long" or p.asset_class.value != "us_equity"
                                          for p in positions):
            raise RuntimeError("Paper account blocked or contains unsupported positions")
        return account, positions, orders, clock

    def bars(self, symbols: list[str], start: datetime, end: datetime):
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        return self.data.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day, start=start, end=end, adjustment=Adjustment.ALL, feed=DataFeed.IEX)).df

    def quote(self, symbol: str) -> tuple[float, datetime]:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest
        asset = self.client.get_asset(symbol)
        if not asset.tradable or asset.asset_class.value != "us_equity":
            raise ValueError("Only tradable US equities are supported")
        quote = self.data.get_stock_latest_quote(StockLatestQuoteRequest(
            symbol_or_symbols=[symbol], feed=DataFeed.IEX))[symbol]
        return float(quote.ask_price), quote.timestamp

    def submit(self, symbol: str, qty: int, price: float, order_id: str):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest
        return self.client.submit_order(order_data=LimitOrderRequest(symbol=symbol, qty=abs(qty),
            side=OrderSide.BUY if qty > 0 else OrderSide.SELL, time_in_force=TimeInForce.DAY,
            limit_price=price, client_order_id=order_id))

    def latest_complete_session(self, timestamp: datetime) -> str:
        from alpaca.trading.requests import GetCalendarRequest
        from zoneinfo import ZoneInfo
        local = timestamp.astimezone(ZoneInfo("America/New_York"))
        calendar = self.client.get_calendar(GetCalendarRequest(
            start=local.date().isoformat(), end=local.date().isoformat()))
        if not calendar:
            raise ValueError("Current date has no market session")
        earlier = (local - timedelta(days=10)).date()
        previous = (local - timedelta(days=1)).date()
        past = self.client.get_calendar(GetCalendarRequest(start=earlier.isoformat(), end=previous.isoformat()))
        return past[-1].date.isoformat()

    def lookup(self, order_id: str):
        return self.client.get_order_by_client_id(order_id)

    def cancel(self, order_id: str) -> None:
        self.client.cancel_order_by_id(self.lookup(order_id).id)

class Execution:
    def __init__(self, path: str, limits: Limits, broker: Broker | None = None,
                 paper_enabled: bool = False) -> None:
        self.path, self.limits, self.broker, self.paper_enabled = path, limits, broker, paper_enabled
        with self.database() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY, account TEXT, day TEXT, symbol TEXT, qty INTEGER,
                price REAL, status TEXT, filled REAL DEFAULT 0, average REAL)""")
            db.execute("CREATE TABLE IF NOT EXISTS peaks (account TEXT PRIMARY KEY, equity REAL)")

    @contextmanager
    def database(self):
        with closing(sqlite3.connect(self.path)) as database, database:
            yield database

    def journal(self) -> list[dict]:
        with self.database() as database:
            database.row_factory = sqlite3.Row
            return [dict(row) for row in database.execute(
                "SELECT * FROM orders ORDER BY rowid DESC LIMIT 50")]

    def reconcile(self) -> list[tuple]:
        with self.database() as db:
            for (order_id,) in db.execute("""SELECT id FROM orders WHERE status NOT IN
                    ('filled','canceled','expired','rejected','dry-run')""").fetchall():
                self.record(db, order_id, self.broker.lookup(order_id))
            return db.execute("SELECT * FROM orders ORDER BY rowid DESC").fetchall()

    @staticmethod
    def record(db: sqlite3.Connection, order_id: str, order) -> None:
        db.execute("UPDATE orders SET status=?,filled=?,average=? WHERE id=?",
                   (getattr(order.status, "value", order.status), float(order.filled_qty or 0),
                    float(order.filled_avg_price) if order.filled_avg_price else None, order_id))

    def dry_run(self, symbol: str, target: float, price: float, atr: float,
                date: str) -> str:
        account = Account(100000, 100000, 0, 100000, 100000, 100000)
        delta, reason = approve(self.limits, account, 0, int(target * account.equity / price), price, atr)
        if delta:
            with self.database() as db:
                db.execute("INSERT OR IGNORE INTO orders(id,account,day,symbol,qty,price,status) VALUES(?,?,?,?,?,?,?)",
                           (f"dry-{symbol}-{date}", "dry", date, symbol, delta, price, "dry-run"))
        return f"{symbol}: {delta} shares proposed ({reason})"

    def paper(self, symbol: str, target: float, atr: float, signal_date: str, *, armed: bool,
              sectors: dict[str, str] | None = None) -> str:
        if not self.paper_enabled or not armed or self.broker is None:
            raise RuntimeError("Paper submissions require configuration and explicit arming")
        self.reconcile()
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            account, positions, orders, clock = self.broker.snapshot()
            account_id = str(account.id)
            day = str(clock.timestamp.date())
            if not clock.is_open or signal_date != self.broker.latest_complete_session(clock.timestamp):
                return "Paper trading requires an open market and the previous completed session"
            if orders or db.execute("""SELECT 1 FROM orders WHERE account=? AND status NOT IN
                    ('filled','canceled','expired','rejected')""", (account_id,)).fetchone():
                return "Pending order or unresolved submission"
            price, timestamp = self.broker.quote(symbol)
            age = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600
            if price < 1 or not 0 <= age <= min(1 / 60, self.limits.max_age_hours):
                return "Invalid or stale paper quote"
            price = round(price, 2)
            equity = float(account.equity)
            db.execute("""INSERT INTO peaks VALUES(?,?) ON CONFLICT(account)
                       DO UPDATE SET equity=max(equity,excluded.equity)""", (account_id, equity))
            peak = db.execute("SELECT equity FROM peaks WHERE account=?", (account_id,)).fetchone()[0]
            cash = float(account.cash)
            holdings = {p.symbol: p for p in positions}
            held = float(holdings[symbol].qty) if symbol in holdings else 0.0
            if not held.is_integer():
                return "Fractional positions require manual reconciliation"
            held = int(held)
            exposure = sum(float(p.market_value) for p in positions)
            sector = (sectors or {}).get(symbol)
            sector_value = (sum(float(p.market_value) for p in positions if sectors.get(p.symbol) == sector)
                            if sector else None)
            count = db.execute("SELECT count(*) FROM orders WHERE account=? AND day=?",
                               (account_id, day)).fetchone()[0]
            order_id = "v2-" + hashlib.sha256(f"{account_id}:{symbol}:{signal_date}".encode()).hexdigest()[:32]
            if db.execute("SELECT 1 FROM orders WHERE id=?", (order_id,)).fetchone():
                return "Signal already processed"
            delta, reason = approve(self.limits, Account(equity, cash, exposure, peak,
                float(account.last_equity), float(account.buying_power)), held,
                int(target * equity / price) - held, price, atr,
                age_hours=age, orders=count, market_open=clock.is_open, sector_value=sector_value)
            if not delta:
                return reason
            db.execute("INSERT INTO orders(id,account,day,symbol,qty,price,status) VALUES(?,?,?,?,?,?,?)",
                       (order_id, account_id, day, symbol, delta, price, "unknown"))
        # Commit the intent before network I/O. Timeouts leave a blocked, reconcilable intent.
        order = self.broker.submit(symbol, delta, price, order_id)
        with self.database() as db:
            self.record(db, order_id, order)
        return f"{symbol} {delta} shares; {order.status}; {order_id}"
