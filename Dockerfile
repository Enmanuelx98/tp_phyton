FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
# Exponemos el puerto que usará FastAPI

EXPOSE 80

CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "80"]
