FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy app and pipelines
COPY ./app ./app
COPY ./Pipelines ./Pipelines
COPY ./helpers ./helpers

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]   