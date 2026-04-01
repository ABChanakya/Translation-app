# Dockerfile for Manga Translation Developer Kit
# Builds a self-contained image to run the web UI and examples

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies commonly required by the project
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ffmpeg \
       git \
       libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Create a non-root user and give ownership of /app
RUN useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000
ENV PORT=5000

# Default command: run the web UI (Flask/Streamlit app at web/app.py)
# If you prefer to run a different entrypoint, override CMD when running container
CMD ["python", "web/app.py"]
