FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Force CPU-only wheels for the exact PyTorch series you need
# (2.0.1 shown here)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

COPY ./src /app/src
COPY ./models /app/models

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=5s --timeout=3s \
  CMD curl -fsS http://localhost:8000/ || exit 1

EXPOSE 8000
CMD ["gunicorn","-b","0.0.0.0:8000","src.flask_server:create_app()"]
