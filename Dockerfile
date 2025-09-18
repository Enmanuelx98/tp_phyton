# Usamos una imagen base de Python 3.10
FROM python:3.10-slim

# Evitamos prompts durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Actualizamos el sistema e instalamos dependencias necesarias para OpenCV y MediaPipe
RUN apt-get update && \
    apt-get install -y build-essential cmake libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Actualizamos pip y herramientas de instalación de paquetes
RUN pip install --upgrade pip setuptools wheel

# Creamos directorio de trabajo
WORKDIR /app

# Copiamos archivos del proyecto al contenedor
COPY . /app

# Instalamos dependencias Python
RUN pip install fastapi uvicorn keras numpy opencv-python mediapipe

# Exponemos el puerto que usará FastAPI
EXPOSE 5000

# Comando para iniciar la aplicación
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "5000"]
