"""DexteraAI CLI entry point.

Usage: python dextera.py <command> [options]
   or: python __main__.py <command> [options]
"""
from dextera import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[DexteraAI ERROR] {e}")
        exit(1)
