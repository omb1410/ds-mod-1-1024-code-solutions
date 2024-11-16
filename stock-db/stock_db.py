# -*- coding: utf-8 -*-
"""stock_db.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/129cW8IpmJnkEmg6cEaI5m94mC8cCvz5-
"""

import sqlite3
import pandas as pd

conn = sqlite3.connect('/content/stocks.sqlite')

tables = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")

tables.fetchall()

conn.execute("""CREATE TABLE stock_MSFT AS
                SELECT *
                FROM STOCK_DATA
                WHERE Symbol = 'MSFT'
              """)

conn.execute("""CREATE TABLE stock_AAPL AS
                SELECT *
                FROM STOCK_DATA
                WHERE Symbol = 'AAPL'
              """)

conn.execute("SELECT * FROM stock_MSFT").fetchall()

conn.execute("SELECT * FROM stock_AAPL").fetchall()

MSFT_cols = conn.execute("PRAGMA table_info(stock_MSFT)").fetchall()
MSFT_cols

MSFT_cols = [elem[1] for elem in MSFT_cols]
MSFT_cols

AAPL_cols = conn.execute("PRAGMA table_info(stock_AAPL)").fetchall()
AAPL_cols

AAPL_cols = [elem[1] for elem in AAPL_cols]
AAPL_cols

MSFT = conn.execute(""" SELECT *
                FROM stock_MSFT
                WHERE Date = (SELECT MAX(Date) FROM stock_MSFT)
                OR Date = (SELECT MIN(Date) FROM stock_MSFT)
""").fetchall()
MSFT

AAPL = conn.execute(""" SELECT *
                FROM stock_AAPL
                WHERE Date = (SELECT MAX(Date) FROM stock_AAPL)
                OR Date = (SELECT MIN(Date) FROM stock_AAPL)
""").fetchall()
AAPL

MSFT_min_max = pd.DataFrame(MSFT, columns = MSFT_cols)
MSFT_min_max

AAPL_min_max = pd.DataFrame(AAPL, columns = AAPL_cols)
AAPL_min_max

MSFT2 = conn.execute(""" SELECT *
                FROM stock_MSFT
                HAVING Open > 50
                    """).fetchall()
MSFT2

AAPL2 = conn.execute(""" SELECT *
                FROM stock_AAPL
                HAVING Open > 50
                    """).fetchall()
AAPL2

MSFT_val_fifty = pd.DataFrame(MSFT2, columns = MSFT_cols)
MSFT_val_fifty

AAPL_val_fifty = pd.DataFrame(AAPL2, columns = AAPL_cols)
AAPL_val_fifty

