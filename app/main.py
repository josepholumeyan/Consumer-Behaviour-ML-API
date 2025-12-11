from typing import List,Union
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pandas as pd
from app.pipeline_wrapper import ModelPipeline
from app.validate import validate_input_data

app = FastAPI(title="Sales Prediction API")


quantity_pipeline = ModelPipeline(pipeline_path="Pipelines/pipeline_Quantity.pkl")

class SalesInput(BaseModel):
    data: Union[dict,List[dict]]

def make_prediction(pipeline: ModelPipeline, input_data:dict):
    try:
        df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else pd.DataFrame(input_data)
        validated_df = validate_input_data(
            df,
            expected_columns=pipeline.get_expected_columns(),
            numeric_columns=pipeline.numeric_columns,
            categorical_columns=pipeline.categorical_columns
        )
        prediction = pipeline.predict(validated_df)
        return prediction.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    


@app.post("/predict/quantity")
def predict_quantity(sales_input: SalesInput):
    if not sales_input.data:
        raise HTTPException(status_code=400, detail="Input data is empty.")
    prediction = make_prediction(quantity_pipeline, sales_input.data)
    return {"quantity_prediction": prediction}

