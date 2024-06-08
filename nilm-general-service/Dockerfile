FROM alpine/curl:8.5.0 AS downloader

RUN curl -o models_data.tar.xz "https://cloud.dfki.de/owncloud/index.php/s/rB35AkfBsb8n275/download?path=%2F&files=models_data.tar.xz"

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN useradd -m appuser

COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN pip install gunicorn 

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