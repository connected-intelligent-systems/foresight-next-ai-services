FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN useradd -m appuser

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