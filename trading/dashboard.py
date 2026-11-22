"""Research dashboard and explicit offline, dry-run, and paper entry points."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from .analytics import attribution, metrics
from .data import load, synthetic
from .execution import AlpacaPaper, Execution
from .research import Research
from .strategies import REASONS

def config_from(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def compare(research: Research) -> tuple[dict, pd.DataFrame]:
    results = {}
    for method in ("hold", *research.config.get("strategies", REASONS), "equal", "rules", "ml"):
        curve, fills = research.run(method)
        results[method] = metrics(curve, fills, research.config.get("initial_cash", 100000))
        results[method]["attribution"] = attribution(fills)
    return results, curve

def make_research(args, config: dict) -> Research:
    if not args.data and not args.synthetic:
        raise ValueError("Specify --data or explicitly select --synthetic")
    research = Research(load(args.data) if args.data else synthetic(), config)
    research.train()
    return research

def session(research: Research, args, *, armed: bool = False) -> dict:
    mode = args.mode
    if mode == "paper" and (args.synthetic or not research.config.get("allow_paper_orders")
                            or not armed):
        raise ValueError("Paper requires real historical data, configuration and explicit arming")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
    broker = AlpacaPaper(key, secret) if mode == "paper" and key and secret else None
    if mode == "paper" and broker is None:
        raise ValueError("Paper credentials are required")
    engine = Execution(args.journal, research.limits, broker,
                       research.config.get("allow_paper_orders", False)) if mode != "offline" else None
    output = {"mode": mode, "source": "SYNTHETIC FIXTURE" if args.synthetic else str(args.data),
              "trained_at": research.selector.trained_at, "validation_mae": research.selector.validation_mae,
              "market_regime": research.market.iloc[-1],
              "market_ml_regime": research.selector.classify(
                  research.frames[research.config.get("benchmark", "SPY")].iloc[[-1]]), "stocks": {}}
    if broker:
        account, positions, orders, _ = broker.snapshot()
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
        if mode == "paper":
            detail["execution"] = engine.paper(symbol, target, atr, str(date.date()),
                                               armed=armed, sectors=research.config.get("sectors"))
    if engine:
        output["journal"] = engine.journal()
    return output

def dashboard(args) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from concurrent.futures import ThreadPoolExecutor

    root = tk.Tk()
    root.title("Trading Bot V2")
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
    display = tk.Text(root, wrap="word")
    display.pack(fill="both", expand=True)
    pool = ThreadPoolExecutor(max_workers=1)
    state = {"running": False, "busy": False, "context": None}

    def show(value) -> None:
        display.delete("1.0", "end")
        display.insert("end", value if isinstance(value, str) else json.dumps(value, indent=2))

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
            research = make_research(selected, options)
            if backtest:
                results, curve = compare(research)
                return {"source": "SYNTHETIC FIXTURE" if selected.synthetic else selected.data,
                        "results": results}, curve.equity.tolist()
            return session(research, selected, armed=armed), None

        def done(result) -> None:
            output, equity = result
            state["context"] = output
            show(output)
            if equity:
                graph.delete("all")
                width = max(graph.winfo_width() - 20, 100)
                span = max(max(equity) - min(equity), 1)
                points = [(10 + i * width / (len(equity) - 1),
                           120 - (value - min(equity)) * 105 / span)
                          for i, value in enumerate(equity)]
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

    def ask() -> None:
        from openai import OpenAI
        prompt = question.get()
        context = json.dumps(state["context"])
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            show("Set OPENAI_API_KEY to use the optional research assistant.")
            return
        execute(lambda: OpenAI(api_key=key).chat.completions.create(model="gpt-5-mini", messages=[
            {"role": "system", "content": "Explain the supplied portfolio and quantitative results. You have no tools. Context: " + context},
            {"role": "user", "content": prompt}]).choices[0].message.content, show)

    for label, command in (("Start", start), ("Stop", lambda: state.update(running=False)),
                           ("Backtest", lambda: action(backtest=True)), ("Run one paper cycle", paper),
                           ("Ask research assistant", ask)):
        ttk.Button(controls, text=label, command=command).pack(side="left")
    show("Ready. Choose a mode and click Start, or run a backtest.")
    root.after(60000, tick)
    root.mainloop()
    pool.shutdown(wait=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dashboard", "backtest", "train", "run", "download"))
    parser.add_argument("--config", default="config.example.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--mode", choices=("offline", "dry-run", "paper"), default="dry-run")
    parser.add_argument("--arm-paper", action="store_true")
    parser.add_argument("--journal", default="orders.sqlite3")
    parser.add_argument("--output", default="bars.csv")
    args = parser.parse_args()
    if args.command == "dashboard":
        dashboard(args)
        return
    config = config_from(args.config)
    if args.command == "download":
        symbols = config.get("symbols", [config.get("benchmark", "SPY")])
        symbols = sorted(set(symbols + [config.get("benchmark", "SPY")]))
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        adapter = AlpacaPaper(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
        frame = adapter.bars(symbols, end - timedelta(days=1500), end).reset_index()
        frame.rename(columns={"timestamp": "date"}).to_csv(args.output, index=False)
        return
    research = make_research(args, config)
    if args.command == "backtest":
        results, _ = compare(research)
        result = {"source": "SYNTHETIC FIXTURE" if args.synthetic else args.data,
                  "test_from": str(research.dates[research.cutoff].date()), "results": results}
    elif args.command == "train":
        result = {"trained_at": research.selector.trained_at,
                  "validation_mae": research.selector.validation_mae,
                  "source": "SYNTHETIC FIXTURE" if args.synthetic else args.data}
    else:
        result = session(research, args, armed=args.arm_paper)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
