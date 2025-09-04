# Use Python 3.11 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libxss1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY api_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Create a modified requirements file without problematic packages
RUN sed -e '/^tensorflow-addons==/d' -e '/^onnx-tf==/d' requirements.txt > requirements_modified.txt

# Install modified requirements
RUN pip install --no-cache-dir -r requirements_modified.txt

RUN pip install --no-cache-dir -r api_requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Create volume for persistent data
VOLUME ["/app/data"]

# Expose port
EXPOSE 5002

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=api.py

# Run the application
CMD ["python", "api.py"]