FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN useradd -m appuser

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r ./requirements.txt
RUN pip3 install gunicorn 

COPY ./src /app/src
COPY ./models /app/models

RUN chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=5s --timeout=3s \
  CMD curl -f http://localhost:8000/ || exit 1

CMD [ "gunicorn", "-b", "0.0.0.0", "src.flask_server:create_app()" ]

EXPOSE 8000
