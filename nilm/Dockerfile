FROM alpine/curl:8.5.0 AS downloader

RUN curl -o models_data.tar.xz "https://cloud.dfki.de/owncloud/index.php/s/rB35AkfBsb8n275/download?path=%2F&files=models_data.tar.xz"

FROM python:3.10-slim-bookworm

WORKDIR /app

RUN useradd -m appuser

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 xz-utils && \
    rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0 --index-url ${TORCH_INDEX_URL}
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY --from=downloader /models_data.tar.xz /app/models_data.tar.xz
RUN tar -xvf /app/models_data.tar.xz -C /app && \
    rm /app/models_data.tar.xz

COPY ./src /app/src
COPY ./settings.json /app/settings.json

RUN chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=5s --timeout=3s \
  CMD curl -f http://localhost:8000/ || exit 1

CMD [ "gunicorn", "-b", "0.0.0.0", "src.app:create_app()" ]

EXPOSE 8000
