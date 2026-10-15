# ==============================================================================
#  Raspberry Pi 5 ARM64 Dockerfile for Truck Fatigue Model
# ==============================================================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system runtime libraries for OpenCV and Video4Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY models/ ./models/
COPY src/ ./src/
COPY app.py edge_runner.py ./

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
