import pandas as pd

def validate_input_data(df: pd.DataFrame, expected_columns=None,numeric_columns=None, categorical_columns=None,fill_defaults: dict =None):
    """
    Validate the input DataFrame to ensure it contains the expected columns
    and that the data types are correct.

    Parameters:
    - df: pd.DataFrame - The input data to validate.
    - expected_columns: List[str] - List of expected column names.
    - numeric_columns: List[str] - List of columns that should be numeric.
    - categorical_columns: List[str] - List of columns that should be categorical.
    - fill_defaults: dict - A dictionary specifying default values for missing columns.

    Returns:
    - pd.DataFrame - The validated and possibly modified DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    if expected_columns:
        missing =[c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                raise ValueError(f"Expected numeric column '{col}' is missing from DataFrame.")
    if categorical_columns:
        for col in categorical_columns:
            if col in df.columns:
                if not pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype('category')
            else:
                raise ValueError(f"Expected categorical column '{col}' is missing from DataFrame.")
    
    if fill_defaults:
        for col, default in fill_defaults.items():
            if col not in df.columns:
                df[col] = default.fillna(default)
    return df