"""Part A task 6 - run the seven queries in sql/queries.sql against data/instamart.db and report results."""
import re
import sqlite3

import pandas as pd

from common import DB, SQL

text = SQL.read_text()
parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
assert list(queries) == [f"Q{i}" for i in range(1, 8)]

res = {}
with sqlite3.connect(DB) as con:
    for q, sql in queries.items():
        res[q] = pd.read_sql_query(sql, con)

r = res["Q1"]; print(f"Q1 customers: {len(r)}")
r = res["Q2"]; top = r.iloc[0]
print(f"Q2 rows: {len(r)}; sum of revenues: {r['revenue'].sum():,.2f}; largest cell: {top['city']} / {top['category']} {top['revenue']:,.2f}")
print(f"Q3 products above average price: {len(res['Q3'])}")
print(f"Q4 customers without a completed order: {len(res['Q4'])}")
r = res["Q5"]; print(f"Q5 customers above average: {len(r)}; average {r['avg_total'].iloc[0]:,.2f}")
r = res["Q6"]
print("Q6 top customer per city:")
print(r[r["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
diff = r[r["rnk"] != r["row_num"]]
print(f"Q6 rows where RANK() != ROW_NUMBER(): {len(diff)}")
r = res["Q7"]; print(f"Q7 daily rows: {len(r)}; last cumulative: {r['cumulative_revenue'].iloc[-1]:,.2f}")
