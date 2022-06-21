FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

COPY "requirements.txt" .
COPY "fhv_tripdata_2021-04.parquet" .
COPY "homework-4.py" .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


CMD ["python", "homework-4.py", "2021", "04"]


