import helpers.model_trainer as mt
import joblib


preprocessor_Quantity = joblib.load("models/preprocessor_Quantity.pkl")
Quantity_model= joblib.load("models/best_model_Quantity.pkl")

pipeline_Quantity = mt.Pipelines(model=Quantity_model, preprocessor=preprocessor_Quantity)

joblib.dump(pipeline_Quantity, "Pipelines/pipeline_Quantity.pkl")


print("Pipelines created and saved successfully.")

