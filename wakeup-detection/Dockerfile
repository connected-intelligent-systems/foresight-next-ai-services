ARG BASE_IMAGE=tensorflow/tensorflow:2.2.0-gpu
FROM $BASE_IMAGE

WORKDIR /app

RUN useradd -m appuser

COPY /requirements.txt /app/requirements.txt
RUN pip3 install -r ./requirements.txt
RUN pip3 install gunicorn 

COPY ./src /app/src
COPY ./res /app/res

RUN chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=5s --timeout=3s \
  CMD curl -f http://localhost:8000/ || exit 1

CMD [ "gunicorn", "-b", "0.0.0.0", "src.server.flask_server:create_app()" ]

EXPOSE 8000