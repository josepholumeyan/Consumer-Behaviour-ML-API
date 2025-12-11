from sklearn.ensemble import RandomForestRegressor
import helpers.model_trainer as mt
import pandas as pd
import joblib
import xgboost as xgb

# Load the processed data and preprocessors
monthly_sales=pd.read_csv("data/processed/monthly_sales_with_engineered_features.csv", encoding="latin1")
preprocessor_Quantity = joblib.load("models/preprocessor_Quantity.pkl")
monthly_sales=monthly_sales.drop(columns=["Description","Revenue"])
monthly_sales["YearMonth"]= monthly_sales["YearMonth"].str[-2:]

# spiltting features and targets
X_Quantity = monthly_sales.drop(columns=["Quantity"])
y_Quantity = monthly_sales["Quantity"]

# Apply preprocessing
X_Quantity = preprocessor_Quantity.transform(X_Quantity)

# Train and evaluate models 
def train_and_evaluate_models(X, y, target_name, preprocessor):
    xgb_model = mt.model_training(model_name=f"XGBoost for {target_name}")
    rf_model = mt.model_training(model_name=f"Random Forest for {target_name}")

    xgb_model.split(X, y).fit_xgboost()
    rf_model.split(X, y).fit_rf()

    models = [xgb_model, rf_model]

    # Evaluate all models
    results = []
    for model in models:
        evaluation = model.eval_model()
        results.append(evaluation)

    # select best model
    best_model_info = max(results, key=lambda x: x['R2'])
    best_model = best_model_info['Model']
    print(f"Best model for {target_name}: {best_model.model_name} with R2 Score: {best_model_info['R2']}")

    # check reliability
    if isinstance(best_model.model, (RandomForestRegressor, xgb.XGBRegressor)):
        best_model.check_relia(preprocessor)

    # Save the best model
    joblib.dump(best_model, f"models/best_model_{target_name}.pkl")
    print(f"Best model saved to 'models/best_model_{target_name}.pkl'")


train_and_evaluate_models(X_Quantity, y_Quantity, "Quantity", preprocessor_Quantity)
