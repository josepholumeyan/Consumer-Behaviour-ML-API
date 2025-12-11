import pandas as pd
import helpers.DA_helper as dah

chunksize = 50000  
chunk_list=[]

for chunk in pd.read_csv("data/raw/data.csv", chunksize=chunksize,encoding="latin1"):
    chunk["InvoiceDate"] = pd.to_datetime(chunk["InvoiceDate"], errors="coerce")
    chunk=chunk.dropna(subset=["InvoiceDate"])
    chunk["YearMonth"] = chunk["InvoiceDate"].dt.to_period("M")
    monthly_sales = chunk.groupby(["StockCode","YearMonth","Description"]).agg({"Quantity": "sum", "UnitPrice": "sum"}).reset_index()
    chunk_list.append(monthly_sales)


df=pd.concat(chunk_list, ignore_index=True)
df=df.sort_values(["StockCode","YearMonth"]).reset_index(drop=True)
print(f"Dataframe shape after ingestion: {df.shape}")
print(df)
output_path="data/raw/monthly_sales.csv"
df.to_csv(output_path, index=False) 
print(f"Monthly sales data saved to {output_path}")