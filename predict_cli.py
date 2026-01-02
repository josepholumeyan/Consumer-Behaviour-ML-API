import pandas as pd
from app.pipeline_wrapper import ModelPipeline
from app.validate import validate_input_data

# Load pipeline
pipeline = ModelPipeline("Pipelines/pipeline_Quantity.pkl")

print("\n=== Interactive Quantity Predictor ===")
print("Type values when prompted. Press Ctrl+C to exit.\n")

while True:
    try:
        row = {}

        for col in pipeline.get_expected_columns():
            val = input(f"{col}: ")

            # convert numbers
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except ValueError:
                pass

            row[col] = val

        df = pd.DataFrame([row])

        df_valid = validate_input_data(
            df,
            expected_columns=pipeline.get_expected_columns(),
            numeric_columns=pipeline.numeric_columns,
            categorical_columns=pipeline.categorical_columns
        )

        pred = pipeline.predict(df_valid)[0]
        print(f"\n Predicted Quantity: {round(pred, 2)}\n")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except Exception as e:
        print(f"\n Error: {e}\n")