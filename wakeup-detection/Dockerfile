ARG BASE_IMAGE=python:3.10-slim-bookworm
FROM $BASE_IMAGE

WORKDIR /app

RUN useradd -m appuser

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY /requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r ./requirements.txt gunicorn

COPY ./src /app/src
COPY ./res /app/res

RUN chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=5s --timeout=3s \
  CMD curl -f http://localhost:8000/ || exit 1

CMD [ "gunicorn", "-b", "0.0.0.0", "src.server.flask_server:create_app()" ]

EXPOSE 8000
