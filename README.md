# Consumer Behaviour ML

![Python](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.124.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-orange)

A machine learning project for predicting **product quantity sold** using historical sales data. Includes **preprocessing pipelines, XGBoost & Random Forest models, input validation, and a FastAPI deployment**.

---

## Project Overview

- Predicts quantity sold based on historical **quantity, unit price, revenue**, and product info.  
- Preprocessing + modeling packaged into reusable **pipelines**.  
- API supports **single and batch predictions**.  

---

## Features

- **Historical Features:** Quantity, UnitPrice, Revenue (t-1, t-2)  
- **Aggregate Features:** Quantity_avg, UnitPrice_avg, Revenue_avg  
- **Temporal Feature:** Month (1–12)  
- **Product Feature:** StockCode  

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/josepholumeyan/Consumer-Behaviour-ML-API
cd Consumer-Behaviour-ML
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```


## Running the API

### Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

```bash
POST /predict/quantity — Predict quantity sold
```

### Example Input
``` bash
Single Row

{
  "data": {
    "StockCode": 22879,
    "Month": 7,
    "Quantity_t-1": 50,
    "Quantity_t-2": 45,
    "UnitPrice": 20.5,
    "UnitPrice_t-1": 20.0,
    "UnitPrice_t-2": 19.8,
    "Revenue_t-1": 1025,
    "Revenue_t-2": 900,
    "Quantity_avg": 47.5,
    "UnitPrice_avg": 20.1,
    "Revenue_avg": 962.5
  }
}

Batch Input

{
  "data": [
    {
      "StockCode": 22879,
      "Month": 7,
      "Quantity_t-1": 50,
      "Quantity_t-2": 45,
      "UnitPrice": 20.5,
      "UnitPrice_t-1": 20.0,
      "UnitPrice_t-2": 19.8,
      "Revenue_t-1": 1025,
      "Revenue_t-2": 900,
      "Quantity_avg": 47.5,
      "UnitPrice_avg": 20.1,
      "Revenue_avg": 962.5
    },
    {
      "StockCode": 23084,
      "Month": 7,
      "Quantity_t-1": 30,
      "Quantity_t-2": 28,
      "UnitPrice": 15.0,
      "UnitPrice_t-1": 14.8,
      "UnitPrice_t-2": 14.5,
      "Revenue_t-1": 450,
      "Revenue_t-2": 420,
      "Quantity_avg": 29,
      "UnitPrice_avg": 14.76,
      "Revenue_avg": 435
    }
  ]
}
```

### Example Usage
``` bash
import requests

url = "http://127.0.0.1:8000/predict/quantity"
payload = {"data": {...}}  # single dict or list of dicts
response = requests.post(url, json=payload)
print(response.json())
```

---

## Model Performance (Quantity)

|Model	|R²	|MAE	|RMSE|
|-----------|:-----------:|:--------:|---------:|
|XGBoost	|0.71	|50.38	|135|
|Random Forest	|0.596	|56	|160|


### Top Features:
- Quantity_avg(0.12) 
- Revenue_avg
- StockCode_XXXX 
- Quantity_t-1 
- UnitPrice_avg


---

## Deployment

### Docker Build and Run:
```bash
docker build -t consumer-behaviour-ml .
docker run -p 8000:8000 consumer-behaviour-ml
```
**Deployable on GCP, AWS, or any cloud platform supporting Docker.**


---

## Notes

- Only quantity prediction is exposed; revenue can be derived as UnitPrice x Quantity.

- API validates missing or incorrect columns with descriptive errors.

- Supports single-row and batch predictions seamlessly.



---

## Folder Structure
```bash
Consumer-Behaviour-ML/
│
├── app/
│   ├── main.py
│   ├── validate.py
│   └── pipeline_wrapper.py
│
├── data/
|   ├── raw/(raw datasets)
│   └── processed/(processed datasets)
│
├── helpers/
|   ├── DA_helper.py
|   └── model_trainer.py
|
├── models/
│   └── ... (trained models & preprocessors)
│
├── pipelines/
│   └── ... (pipeline objects)
│
├── stage_1.py ... stage_5.py
├── requirements.txt
├── Dockerfile
└── README.md
```


