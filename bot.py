import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import alpaca_trade_api as tradeapi
from openai import OpenAI


DATA_FILE = "equities.json"

key = os.environ.get("ALPACA_API_KEY")
secret_key = os.environ.get("ALPACA_API_SECRET")
openai_api_key = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://paper-api.alpaca.markets/"

api = None
if key and secret_key:
    api = tradeapi.REST(key, secret_key, BASE_URL, api_version="v2")

openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

def fetch_portfolio():
    if api is None:
        return []
    positions = api.list_positions()
    portfolio = []
    for pos in positions:
        portfolio.append({
            'symbol': pos.symbol,
            'qty': pos.qty,
            'entry_price': pos.avg_entry_price,
            'current_price': pos.current_price,
            'unrealized_pl': pos.unrealized_pl,
            'side': pos.side
        })
    return portfolio

def fetch_open_orders():
    if api is None:
        return []
    orders = api.list_orders(status='open')
    open_orders = []
    for order in orders:
        open_orders.append({
            'symbol': order.symbol,
            'qty': order.qty,
            'limit_price': order.limit_price,
            'side': order.side
        })
    return open_orders

def fetch_mock_api(symbol):
    if api is None:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
    trade = api.get_latest_trade(symbol)
    return {"price": float(trade.price)}

def chatgpt_response(message):
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not set")

    portfolio_data = fetch_portfolio()
    open_orders = fetch_open_orders()

    pre_prompt = f"""
    You are an AI Portfolio Manager responsible for analyzing my portfolio.
    Evaluate risk exposure, open limit orders, diversification, trade adjustments,
    market outlook, and risk-management strategies.

    Here is my portfolio: {portfolio_data}

    Here are my open orders: {open_orders}

    Answer this question with priority given to that background: {message}
    """

    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": pre_prompt}]
    )
    return response.choices[0].message.content

class TradingBotGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Trading Bot")
        self.equities = self.load_equities()
        self.system_running = False

        self.form_frame = tk.Frame(root)
        self.form_frame.pack(pady=10)

        # Form to add a new equity to our trading bot
        tk.Label(self.form_frame, text="Symbol:").grid(row=0, column=0)
        self.symbol_entry = tk.Entry(self.form_frame)
        self.symbol_entry.grid(row=0, column=1)

        tk.Label(self.form_frame, text="Levels:").grid(row=0, column=2)
        self.levels_entry = tk.Entry(self.form_frame)
        self.levels_entry.grid(row=0, column=3)

        tk.Label(self.form_frame, text="Drawdown%:").grid(row=0, column=4)
        self.drawdown_entry = tk.Entry(self.form_frame)
        self.drawdown_entry.grid(row=0, column=5)

        self.add_button = tk.Button(self.form_frame, text="Add Equity", command=self.add_equity)
        self.add_button.grid(row=0, column=6)

        # Table to track the traded equities
        self.tree = ttk.Treeview(root, columns=("Symbol", "Position", "Entry Price", "Levels", "Status"), show='headings')
        for col in ["Symbol", "Position", "Entry Price", "Levels", "Status"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)
        self.tree.pack(pady=10)

        # Buttons to control the bot
        self.toggle_system_button = tk.Button(root, text="Toggle Selected System", command=self.toggle_selected_system)
        self.toggle_system_button.pack(pady=5)

        self.remove_button = tk.Button(root, text="Remove Selected Equity", command=self.remove_selected_equity)
        self.remove_button.pack(pady=5)

        # AI Component
        self.chat_frame = tk.Frame(root)
        self.chat_frame.pack(pady=10)

        self.chat_input = tk.Entry(self.chat_frame, width=50)
        self.chat_input.grid(row=0, column=0, padx=5)

        self.send_button = tk.Button(self.chat_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=1)

        self.chat_output = tk.Text(root, height=5, width=60, state=tk.DISABLED)
        self.chat_output.pack()

        # Load saved data
        self.refresh_table()

        # Auto-refreshing
        self.running = True
        self.root.after(5000, self.auto_update)


    def add_equity(self):
        symbol = self.symbol_entry.get().upper()
        levels = self.symbol_entry.master.winfo_exists() and self.levels_entry.get()
        drawdown = self.drawdown_entry.get()

        if not symbol or not levels.isdigit() or not drawdown.replace('.', '', 1).isdigit():
            messagebox.showerror("Error", "Invalid Input")
            return

        try:
            levels = int(levels)
            drawdown = float(drawdown) / 100
            if levels < 1 or drawdown < 0:
                raise ValueError
            entry_price = fetch_mock_api(symbol)['price']
        except Exception as exc:
            messagebox.showerror("Error", f"Could not add equity: {exc}")
            return

        level_prices = {i + 1: round(entry_price * (1 - drawdown * (i + 1)), 2)
                        for i in range(levels)}

        self.equities[symbol] = {
            "position": 0,
            "entry_price": entry_price,
            "levels": level_prices,
            "drawdown": drawdown,
            "status": "Off"
        }
        self.save_equities()
        self.refresh_table()

    def toggle_selected_system(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "No Equity is Selected")
            return
        
        for item in selected_items:
            symbol = self.tree.item(item)['values'][0]
            self.equities[symbol]['status'] = "On" if self.equities[symbol]['status'] == "Off" else "Off"

        self.save_equities()
        self.refresh_table()
    
    def remove_selected_equity(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "No Equity Selected")
            return
        
        for item in selected_items:
            symbol = self.tree.item(item)['values'][0]
            if symbol in self.equities:
                del self.equities[symbol]
        
        self.save_equities()
        self.refresh_table()

    def send_message(self):
        message = self.chat_input.get()
        if not message:
            return

        try:
            response = chatgpt_response(message)
        except Exception as exc:
            messagebox.showerror("AI Error", str(exc))
            return

        self.chat_output.config(state=tk.NORMAL)
        self.chat_output.insert(tk.END, f"You: {message}\n{response}\n\n")
        self.chat_output.config(state=tk.DISABLED)
        self.chat_input.delete(0, tk.END)

    def fetch_alpaca_data(self, symbol):
        if api is None:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
        barset = api.get_latest_trade(symbol)
        return {"price": float(barset.price)}

    def check_existing_orders(self, symbol, price):
        if api is None:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
        try:
            orders = api.list_orders(status="open", symbols=[symbol])
            return any(order.limit_price is not None and float(order.limit_price) == float(price)
                       for order in orders)
        except Exception as exc:
            messagebox.showerror("API Error", f"Error checking orders: {exc}")
            return False

    def get_max_entry_price(self, symbol):
        if api is None:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET must be set")
        orders = api.list_orders(status="filled", limit=50, symbols=[symbol])
        prices = [float(order.filled_avg_price) for order in orders
                  if order.filled_avg_price is not None and order.symbol == symbol]
        return max(prices) if prices else None

    def trade_systems(self):
        if api is None:
            messagebox.showwarning("Trading Disabled", "Set Alpaca credentials before enabling a system")
            return

        for symbol, data in list(self.equities.items()):
            if data["status"] != "On":
                continue

            try:
                position = api.get_position(symbol)
                entry_price = float(position.avg_entry_price)
                data["position"] = float(position.qty)
            except Exception as exc:
                messagebox.showerror("Position Error",
                                     f"Could not safely inspect {symbol}: {exc}\nNo order was placed")
                continue

            data["entry_price"] = entry_price
            self.save_equities()

        self.refresh_table()
    
    def place_order(self, symbol, price, level):
        if api is None:
            messagebox.showwarning("Trading Disabled", "Set Alpaca credentials before placing orders")
            return

        if -level in self.equities[symbol]["levels"]:
            return

        try:
            api.submit_order(
                symbol=symbol,
                qty=1,
                side="buy",
                type="limit",
                time_in_force="gtc",
                limit_price=str(price)
            )
            self.equities[symbol]["levels"][-level] = price
            del self.equities[symbol]["levels"][level]
            self.save_equities()
        except Exception as exc:
            messagebox.showerror("Order Error", f"Error placing order: {exc}")

    def refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for symbol, data in self.equities.items():
            self.tree.insert("", "end", values=(
                symbol,
                data.get("position", 0),
                data.get("entry_price", 0),
                str(data.get("levels", {})),
                data.get("status", "Off")
            ))

    def auto_update(self):
        if getattr(self, "running", False):
            self.trade_systems()
            self.root.after(5000, self.auto_update)

    def save_equities(self):
        with open(DATA_FILE, "w", encoding="utf-8") as file:
            json.dump(self.equities, file)

    def load_equities(self):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def on_close(self):
        self.running = False
        self.save_equities()
        self.root.destroy()
    
if __name__ == '__main__':
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()







