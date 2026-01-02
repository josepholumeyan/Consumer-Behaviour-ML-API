from typing import List,Union
from fastapi import FastAPI,HTTPException, logger
from pydantic import BaseModel, Field
import pandas as pd
from app.pipeline_wrapper import ModelPipeline
from app.validate import validate_input_data
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Sales Prediction API")

try:
    quantity_pipeline = ModelPipeline(pipeline_path="Pipelines/pipeline_Quantity.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to initialize quantity pipeline: {e}")

class QuantityFeatures(BaseModel):
    StockCode: str = Field(..., example="22879")
    YearMonth: str = Field(..., example="7",description="Month of the year NOT Year and Month (1-12)")
    Quantity_t_1: float = Field(..., example=50)
    Quantity_t_2: float = Field(..., example=45)
    UnitPrice: float = Field(..., example=20.5)
    UnitPrice_t_1: float = Field(..., example=20.0)
    UnitPrice_t_2: float = Field(..., example=19.8)
    Revenue_t_1: float = Field(..., example=1025)
    Revenue_t_2: float = Field(..., example=900)
    Quantity_avg: float = Field(..., example=47.5)
    UnitPrice_avg: float = Field(..., example=20.1)
    Revenue_avg: float = Field(..., example=962.5)


class SalesInput(BaseModel):
    data: Union[QuantityFeatures,List[QuantityFeatures]] 

class QuantityPredictionResponse(BaseModel):
    quantity_prediction: List[float]

def make_prediction(pipeline: ModelPipeline, input_data:dict):
    try:
        df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else pd.DataFrame(input_data)
        logger.info("validated_df: %s", df.describe())
        validated_df = validate_input_data(
            df,
            expected_columns=pipeline.get_expected_columns(),
            numeric_columns=pipeline.numeric_columns,
            categorical_columns=pipeline.categorical_columns
        )
        validated_df = validated_df[pipeline.get_expected_columns()]
        logger.info("validated_df: %s", validated_df.describe())
        prediction = pipeline.predict(validated_df)
        return prediction.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    


@app.post("/predict/quantity", response_model=QuantityPredictionResponse, summary="Predict Quantity Sold", description="Accepts single or batch sales record and return predicted quantity.")

def predict_quantity(sales_input: SalesInput):
    if not sales_input.data:
        raise HTTPException(status_code=400, detail="Input data is empty.")
    prediction = make_prediction(quantity_pipeline, sales_input.data)
    return QuantityPredictionResponse(quantity_prediction=prediction)