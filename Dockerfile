# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by some Python packages
# libpq-dev is needed for PostgreSQL, gcc for compiling C extensions
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
# If requirements don't change, this layer is reused on rebuild
COPY requirements.txt .

# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app folder into the container
COPY app/ .

# Copy the docs folder so test PDFs are available inside the container
COPY docs/ ./docs/

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Start the FastAPI app with uvicorn
# --host 0.0.0.0 allows connections from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]