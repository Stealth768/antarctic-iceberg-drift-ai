# Use lightweight official Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies for spatial/netCDF tools & C compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libnetcdf-dev \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and project data
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .

# Expose FastAPI port
EXPOSE 8000

# Healthcheck to verify engine readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch production Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
