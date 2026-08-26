# Flask + XGBoost prediction API
FROM python:3.11-slim

# XGBoost needs the OpenMP runtime; it is not in python:*-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so edits to app.py don't rebuild this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app.py + the three .pkl artifacts
COPY . .

ENV PORT=5000
EXPOSE 5000

# 2 workers, 120s timeout (each /api/predict runs 22 model.predict calls)
CMD ["sh", "-c", "gunicorn -w 2 -t 120 -b 0.0.0.0:${PORT} app:app"]
