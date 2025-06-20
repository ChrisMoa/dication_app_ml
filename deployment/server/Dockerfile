FROM python:3.9

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch CPU version from correct index
RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install other dependencies
COPY requirements_server.txt .
RUN pip install --no-cache-dir -r requirements_server.txt

# Copy application code and model
COPY gec_server_pytorch.py ./gec_server_pytorch.py
COPY german_gec_mt5/ ./german_gec_mt5/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run server
CMD ["python", "gec_server_pytorch.py"]