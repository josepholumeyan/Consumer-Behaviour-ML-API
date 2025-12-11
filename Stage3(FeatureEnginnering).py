import pandas as pd
import helpers.DA_helper as dah
import joblib
import helpers.model_trainer as mt

monthly_sales=pd.read_csv("data/processed/monthly_sales_with_revenue.csv", encoding="latin1")

lag_cols=["Quantity","UnitPrice","Revenue"]
n_lags=2
monthly_sales.sort_values(["StockCode","YearMonth"], inplace=True)
monthly_sales= dah.add_lag_features(monthly_sales, "StockCode","YearMonth", lag_cols, n_lags)

monthly_sales.dropna(subset=[f"{col}_t-{lag}" for col in lag_cols for lag in range(1, n_lags + 1)], inplace=True)
monthly_sales= dah.add_average_features(monthly_sales, "StockCode", lag_cols)
monthly_sales.to_csv("data/processed/monthly_sales_with_engineered_features.csv", index=False)
print(f"Dataframe shape after feature engineering: {monthly_sales.shape}")

print("========== pipelining================")
monthly_sales=monthly_sales.drop(columns=["Description","Revenue"])
monthly_sales["YearMonth"]= monthly_sales["YearMonth"].str[-2:]

preprocessor_Quantity = mt.build_preprocessor(monthly_sales, target="Quantity")
joblib.dump(preprocessor_Quantity, "models/preprocessor_Quantity.pkl")
