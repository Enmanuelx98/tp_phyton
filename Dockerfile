FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias necesarias para OpenCV y MediaPipe
RUN apt-get update && \
    apt-get install -y ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Render asigna el puerto dinámicamente
ENV PORT=5000
EXPOSE 5000

# Usa la variable de entorno PORT en lugar de un puerto fijo
CMD ["sh", "-c", "uvicorn API:app --host 0.0.0.0 --port $PORT"]
