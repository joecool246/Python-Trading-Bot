"""Trading Bot V4 entry point. Running without arguments opens the demo dashboard."""
import sys
from trading.dashboard import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(("dashboard", "--synthetic"))
    main()
