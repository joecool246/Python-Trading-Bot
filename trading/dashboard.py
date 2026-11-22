"""Research dashboard and explicit offline, dry-run, and paper entry points."""
import argparse
import json
import os
from alpaca.common.exceptions import APIError
from requests.exceptions import RequestException
from pathlib import Path
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from contextlib import closing
from dataclasses import asdict
import sqlite3
from .events import load_events, now, timestamp, unique
import pandas as pd
from .analytics import attribution, daily_returns, metrics, paired_edge
from .data import export_bars, load, symbols_from, synthetic
from .execution import AlpacaPaper, Execution
from .research import Research
from .strategies import REASONS

def config_from(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def compare(research: Research) -> tuple[dict, pd.DataFrame]:
    results, returns = {}, {}
    initial = research.config.get("initial_cash", 100000)
    for method in ("hold", *research.config.get("strategies", REASONS), "equal", "rules", "ml"):
        curve, fills = research.run(method)
        results[method] = metrics(curve, fills, initial)
        results[method]["attribution"] = attribution(fills)
        returns[method] = daily_returns(curve, initial)
    results["ml_comparison"] = {
        name: dict(paired_edge(returns["ml"], series,
                               hurdle=research.config.get("materiality_hurdle", .01)),
                   primary=name == "equal")
        for name, series in returns.items() if name != "ml"}
    return results, curve

class ResearchCache:
    def __init__(self):
        self.entries = {}

    def get(self, args, config: dict, *, deployment: bool) -> Research:
        source = Path(args.data).resolve() if args.data else None
        stat = source.stat() if source else None
        version = (source, stat.st_mtime_ns, stat.st_size) if stat else None
        key = (version, args.synthetic, json.dumps(config, sort_keys=True))
        cached = self.entries.get(deployment)
        if cached is None or cached[0] != key:
            cached = key, make_research(args, config, deployment=deployment)
            self.entries[deployment] = cached
        return cached[1]

def make_research(args, config: dict, *, deployment: bool = False) -> Research:
    if not args.data and not args.synthetic:
        raise ValueError("Specify --data or explicitly select --synthetic")
    research = Research(load(args.data) if args.data else synthetic(), config)
    research.train(deployment=deployment)
    return research

def paper_adapter() -> AlpacaPaper:
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_API_SECRET to use Alpaca paper access.")
    if not key.startswith("PK"):
        raise ValueError("Alpaca access requires a paper API key.")
    return AlpacaPaper(key, secret)

def assistant_context(output: dict | None) -> dict:
    allowed = {"mode", "fit_mode", "market_regime", "market_ml_regime", "risk_limits", "results"}
    return {key: value for key, value in (output or {}).items() if key in allowed}

def session(research: Research, args, *, armed: bool = False) -> dict:
    mode = args.mode
    if mode == "paper" and (args.synthetic or research.fit_mode != "deployment"
                            or research.config.get("allow_paper_orders") is not True or not armed):
        raise ValueError("Paper requires real historical data, configuration and explicit arming")
    broker = paper_adapter() if mode == "paper" else None
    engine = Execution(args.journal, research.limits, broker,
                       research.config.get("allow_paper_orders", False)) if mode != "offline" else None
    output = {"mode": mode, "source": "SYNTHETIC FIXTURE" if args.synthetic else str(args.data),
              "trained_at": research.selector.trained_at, "validation_mae": research.selector.validation_mae,
              "fit_mode": research.fit_mode, "last_training_outcome": str(research.selector.last_label),
              "risk_limits": vars(research.limits),
              "market_regime": research.market.iloc[-1],
              "market_ml_regime": research.selector.classify(
                  research.frames[research.config.get("benchmark", "SPY")].iloc[[-1]]), "stocks": {}}
    if broker:
        account, positions, orders, clock = broker.snapshot()
        if (not clock.is_open or str(research.dates[-1].date())
                != broker.latest_complete_session(clock.timestamp)):
            raise ValueError("Refresh data through the previous completed session while the market is open")
        output["portfolio"] = {"equity": account.equity, "cash": account.cash,
                               "positions": [dict(symbol=p.symbol, qty=p.qty, value=p.market_value,
                                    unrealized=p.unrealized_pl) for p in positions],
                               "open_orders": [str(order.client_order_id) for order in orders]}
    for symbol in research.universe:
        date = research.dates[-1]
        if date not in research.signals[symbol].index:
            output["stocks"][symbol] = {"execution": "missing latest benchmark session"}
            continue
        allocations, explanations = research.decision(symbol, date, "ml")
        atr = float(research.frames[symbol].at[date, "atr"])
        target = sum(allocations.values()) * min(research.limits.position,
                                                research.limits.exposure / len(research.universe))
        detail = {"regime": research.regimes[symbol].loc[date],
                  "ml_regime": research.selector.classify(research.frames[symbol].loc[[date]]),
                  "allocation": allocations,
                  "reasons": explanations, "target": target}
        output["stocks"][symbol] = detail
        if mode == "dry-run" and target:
            detail["execution"] = engine.dry_run(symbol, target,
                float(research.bars[symbol].at[date, "close"]), atr, str(date.date()))
        if mode == "paper" and (research.regimes[symbol].loc[date] == "uncertain"
                                or research.market.loc[date] == "uncertain"):
            detail["execution"] = "Insufficient market history; no order"
        elif mode == "paper":
            detail["execution"] = engine.paper(symbol, target, atr, str(date.date()),
                                               armed=armed, sectors=research.config.get("sectors"),
                                               expected_account=str(account.id))
    if engine:
        output["journal"] = engine.journal()
    return output

def intelligence_view(args) -> dict:
    result = {"kind": "LLM interpretations; prices and execution are independent"}
    if path := getattr(args, "events", None):
        current = now()
        items = [item for item in unique(load_events(path))
                 if item.available <= current and (current - timestamp(item.source.published_at)).days <= 7]
        result["events"] = [
            {"symbol": item.source.symbol, "source": item.source.url, "published": item.source.published_at,
             "available": item.extracted_at, "provider": item.provider, "model": item.model,
             "interpretation": asdict(item.interpretation)}
            for item in sorted(items, key=lambda item: item.available, reverse=True)[:20]]
    if path := getattr(args, "llm_ledger", None):
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            count, cost = db.execute("SELECT count(*), coalesce(sum(cost),0) FROM usage").fetchone()
        result["costs"] = {"requests": count, "estimated_or_reserved_usd": cost}
    if path := getattr(args, "llm_study_report", None):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        result["experiment"] = {key: report[key] for key in ("source", "status", "test_from", "test_to",
            "methods", "paired_llm_edge") if key in report}
    return result

def dashboard(args) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from concurrent.futures import ThreadPoolExecutor

    root = tk.Tk()
    root.title("Trading Bot V4 | Synthetic demo" if args.synthetic else "Trading Bot V4 | Historical data")
    root.geometry("1050x720")
    config = config_from(args.config)
    mode = tk.StringVar(value=args.mode)
    symbols = tk.StringVar(value=",".join(config.get("symbols", [])))
    cap = tk.StringVar(value=str(config.get("risk", {}).get("position", .15)))
    strategies = {name: tk.BooleanVar(value=name in config.get("strategies", REASONS)) for name in REASONS}
    header = ttk.Frame(root)
    header.pack(fill="x")
    ttk.Label(header, text="Execution mode").pack(side="left")
    ttk.Combobox(header, textvariable=mode, values=("offline", "dry-run", "paper"),
                 state="readonly", width=10).pack(side="left")
    ttk.Label(header, text="Symbols").pack(side="left")
    ttk.Entry(header, textvariable=symbols, width=26).pack(side="left")
    ttk.Label(header, text="Max position fraction").pack(side="left")
    ttk.Entry(header, textvariable=cap, width=7).pack(side="left")
    checks = ttk.Frame(root)
    checks.pack(fill="x")
    for name, variable in strategies.items():
        ttk.Checkbutton(checks, text=name, variable=variable).pack(side="left")
    controls = ttk.Frame(root)
    controls.pack(fill="x")
    question = ttk.Entry(root)
    question.pack(fill="x")
    graph = tk.Canvas(root, height=130, background="white")
    graph.pack(fill="x")
    allocations_chart = tk.Canvas(root, height=130, background="white")
    allocations_chart.pack(fill="x")
    display = tk.Text(root, wrap="word")
    display.pack(fill="both", expand=True)
    pool = ThreadPoolExecutor(max_workers=1)
    state = {"running": False, "busy": False, "context": None}
    cache = ResearchCache()

    def show(value) -> None:
        display.delete("1.0", "end")
        display.insert("end", value if isinstance(value, str) else json.dumps(value, indent=2))

    def plot_allocations(groups: dict) -> None:
        allocations_chart.delete("all")
        if not groups:
            return
        palette = dict(zip(REASONS, ("#1963a3", "#22845d", "#8258a5", "#c6722b", "#ac4260", "#47737b")))
        width = max(allocations_chart.winfo_width() - 155, 100)
        allocations_chart.create_text(10, 10, text="ML exposure by prior-close regime", anchor="w")
        for row, (label, weights) in enumerate(groups.items()):
            y = 29 + row * 17
            allocations_chart.create_text(10, y, text=label, anchor="w")
            x = 115.
            for name, weight in sorted(weights.items()):
                next_x = x + width * weight
                if next_x > x:
                    allocations_chart.create_rectangle(x, y - 5, next_x, y + 5, fill=palette[name], width=0)
                x = next_x
            allocations_chart.create_text(x + 4, y, text=f"{sum(weights.values()):.1%}", anchor="w")
        for index, (name, color) in enumerate(palette.items()):
            x = 10 + index * 105
            allocations_chart.create_rectangle(x, 112, x + 10, 122, fill=color, width=0)
            allocations_chart.create_text(x + 13, 117, text=name, anchor="w")

    def execute(work, finished) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        future = pool.submit(work)

        def poll() -> None:
            if not future.done():
                root.after(100, poll)
                return
            state["busy"] = False
            try:
                finished(future.result())
            except (APIError, RequestException, TimeoutError):
                state["running"] = False
                show("Service request failed. Check credentials, access and connectivity. "
                     "If paper execution was armed, reconcile the saved journal before continuing.")
            except Exception as error:
                state["running"] = False
                show(str(error))
        root.after(100, poll)

    def action(backtest: bool = False, armed: bool = False) -> None:
        if state["busy"]:
            return
        try:
            options = dict(config, strategies=[name for name, flag in strategies.items() if flag.get()])
            options["risk"] = dict(config.get("risk", {}), position=float(cap.get()))
            if symbols.get().strip():
                options["symbols"] = [s.strip().upper() for s in symbols.get().split(",") if s.strip()]
            selected = argparse.Namespace(**vars(args))
            selected.mode = mode.get()
        except ValueError as error:
            show(str(error))
            return

        def work():
            research = cache.get(selected, options, deployment=not backtest)
            if backtest:
                results, curve = compare(research)
                return {"source": "SYNTHETIC FIXTURE" if selected.synthetic else selected.data,
                        "results": results}, curve.equity.tolist()
            return session(research, selected, armed=armed), None

        def done(result) -> None:
            output, equity = result
            state["context"] = output
            show(output)
            plot_allocations(output.get("results", {}).get("ml", {}).get("allocation_by_regime", {}))
            if equity:
                graph.delete("all")
                width = max(graph.winfo_width() - 20, 100)
                span = max(max(equity) - min(equity), 1)
                points = [(10 + i * width / (len(equity) - 1),
                           120 - (value - min(equity)) * 105 / span)
                          for i, value in enumerate(equity)]
                if len(points) > 1:
                    graph.create_line(points, fill="#1665ad", width=2)
        execute(work, done)

    def tick() -> None:
        if state["running"] and mode.get() != "paper":
            action()
        root.after(60000, tick)

    def start() -> None:
        if mode.get() == "paper":
            messagebox.showinfo("Paper mode", "Use Run one paper cycle to explicitly arm each cycle.")
            return
        state["running"] = True
        action()

    def paper() -> None:
        if mode.get() == "paper" and messagebox.askyesno("Arm paper cycle", "Submit eligible Alpaca paper orders for this cycle?"):
            action(armed=True)

    def show_intelligence() -> None:
        execute(lambda: intelligence_view(args), show)

    def ask() -> None:
        from openai import OpenAI
        prompt = question.get()
        context = json.dumps(assistant_context(state["context"]))
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            show("Set OPENAI_API_KEY to use the optional research assistant.")
            return
        execute(lambda: OpenAI(api_key=key).chat.completions.create(model="gpt-5-mini", messages=[
            {"role": "system", "content": "Explain the supplied portfolio and quantitative results. You have no tools. Context: " + context},
            {"role": "user", "content": prompt}]).choices[0].message.content, show)

    for label, command in (("Start", start), ("Stop", lambda: state.update(running=False)),
                           ("Backtest", lambda: action(backtest=True)), ("Run one paper cycle", paper),
                           ("Financial intelligence", show_intelligence), ("Ask research assistant", ask)):
        ttk.Button(controls, text=label, command=command).pack(side="left")
    show("Ready. Choose a mode and click Start, or run a backtest.")
    root.after(60000, tick)
    root.mainloop()
    pool.shutdown(wait=False)

def download(config: dict, args, adapter) -> None:
    benchmark = config.get("benchmark", "SPY")
    symbols = list(dict.fromkeys([*symbols_from(config, [benchmark]), benchmark]))
    today = pd.Timestamp(datetime.now(timezone.utc)).normalize()
    end = pd.to_datetime(args.end, utc=True).normalize() if args.end else today
    start = pd.to_datetime(args.start, utc=True).normalize() if args.start else end - timedelta(days=1500)
    if not start < end <= today:
        raise ValueError("Download requires start < end <= today's date; end is exclusive")
    export_bars(adapter, symbols, start.to_pydatetime(), end.to_pydatetime(), args.output,
                feed=args.feed, adjustment=args.adjustment)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dashboard", "backtest", "train", "run", "download", "verify-paper"))
    parser.add_argument("--config", default="config.example.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--mode", choices=("offline", "dry-run", "paper"), default="dry-run")
    parser.add_argument("--arm-paper", action="store_true")
    parser.add_argument("--journal", default="orders.sqlite3")
    parser.add_argument("--output", default="bars.csv")
    parser.add_argument("--start", help="Inclusive download date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Exclusive download date (YYYY-MM-DD)")
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument("--adjustment", choices=("raw", "split", "all"), default="all")
    parser.add_argument("--test-start", help="First holdout session (YYYY-MM-DD)")
    parser.add_argument("--deployment", action="store_true", help="Train through pre-decision history")
    parser.add_argument("--events", help="Read-only financial intelligence JSONL for dashboard")
    parser.add_argument("--llm-ledger", help="Read-only LLM usage ledger for dashboard")
    parser.add_argument("--llm-study-report", help="Offline LLM comparison JSON for dashboard")
    args = parser.parse_args()
    try:
        run_command(args)
    except (APIError, RequestException, TimeoutError):
        parser.exit(1, "Service request failed. Check credentials, access and connectivity. "
                    "If paper execution was armed, reconcile the saved journal before continuing.\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")

def run_command(args: argparse.Namespace) -> None:
    if args.command == "dashboard":
        dashboard(args)
        return
    if args.command == "verify-paper":
        adapter = paper_adapter()
        print(json.dumps(adapter.read_metadata(), indent=2))
        return
    config = config_from(args.config)
    if args.command == "download":
        adapter = paper_adapter()
        download(config, args, adapter)
        return
    if args.test_start:
        config["test_start"] = args.test_start
    research = make_research(args, config, deployment=args.command == "run"
                             or (args.command == "train" and args.deployment))
    if args.command == "backtest":
        results, _ = compare(research)
        result = {"source": "SYNTHETIC FIXTURE" if args.synthetic else args.data,
                  "source_sha256": sha256(Path(args.data).read_bytes()).hexdigest() if args.data else None,
                  "test_from": str(research.dates[research.cutoff].date()),
                  "test_to": str(research.dates[-1].date()), "universe": research.universe,
                  "development_last_label": str(research.selector.last_label.date()),
                  "validation_mae": research.selector.validation_mae, "results": results}
    elif args.command == "train":
        result = {"trained_at": research.selector.trained_at, "fit_mode": research.fit_mode,
                  "last_training_outcome": str(research.selector.last_label.date()),
                  "validation_mae": research.selector.validation_mae,
                  "source": "SYNTHETIC FIXTURE" if args.synthetic else args.data}
    else:
        result = session(research, args, armed=args.arm_paper)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
