import pandas as pd

monthly_sales=pd.read_csv("data/raw/monthly_sales.csv", encoding="latin1")

monthly_sales["Revenue"] = monthly_sales["Quantity"] * monthly_sales["UnitPrice"]

print("thats it")
monthly_sales.sort_values("Revenue", ascending=False, inplace=True)
monthly_sales.drop(monthly_sales[monthly_sales["Description"]=="Manual"].index, inplace=True)
rows_to_drop=[ "POST","DOT","BANK CHARGES"]
monthly_sales=monthly_sales.drop(monthly_sales[monthly_sales["Revenue"]<=0].index)

monthly_sales_top_10=monthly_sales.head(10)
for _, row in monthly_sales_top_10.iterrows():
    if row["YearMonth"][-2:] not in ["11","12"]:
        rows_to_drop.append(row["StockCode"])

month_counts=monthly_sales.groupby("StockCode")["YearMonth"].nunique()
for stock_code, count in month_counts.items():
    if count < 3:
        rows_to_drop.append(stock_code)

for row in rows_to_drop:
    monthly_sales=monthly_sales.drop(monthly_sales[monthly_sales["StockCode"]==row].index)

monthly_sales.to_csv("data/processed/monthly_sales_with_revenue.csv", index=False)
print("Monthly sales with revenue data saved to data/processed/monthly_sales_with_revenue.csv")