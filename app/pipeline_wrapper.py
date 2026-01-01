import joblib
import pandas as pd
from typing import List, Optional

class ModelPipeline:
    def __init__(self, pipeline_path: str=None, model=None, preprocessor=None,expected_columns: Optional[List[str]]=None,numeric_columns: Optional[List[str]]=None,categorical_columns: Optional[List[str]]=None):
        
        if pipeline_path:
            try:
                obj=joblib.load(pipeline_path)
            except Exception as e:
                raise ValueError(f"Failed to load pipeline from {pipeline_path}: {e}")
            if hasattr(obj, "predict"):
                self.pipeline=obj
                self.expected_columns= getattr(obj, "expected_columns", expected_columns)
                self.numeric_columns= getattr(obj, "numeric_columns", numeric_columns)
                self.categorical_columns= getattr(obj, "categorical_columns", categorical_columns)
                self.model= getattr(obj, "model", None)
                self.preprocessor= getattr(obj, "preprocessor", None)
            else:
                raise ValueError("The loaded object is not a valid model pipeline.")
        else:
            if not model or not preprocessor:
                raise ValueError("Both model and preprocessor must be provided if pipeline_path is not given.")
            self.pipeline=None
            self.model=model
            self.preprocessor=preprocessor
            self.expected_columns=expected_columns #automate from preprocessor later
            self.numeric_columns=preprocessor.numeric_columns if hasattr(preprocessor, 'numeric_columns') else numeric_columns
            self.categorical_columns=preprocessor.categorical_columns if hasattr(preprocessor, 'categorical_columns') else categorical_columns
        
    def predict(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
             df = pd.DataFrame([df])
        if self.pipeline is not None:
            return self.pipeline.predict(df)
        X=self.preprocessor.transform(df)
        return self.model.predict(X)
                    
    def get_expected_columns(self):
        return self.expected_columns