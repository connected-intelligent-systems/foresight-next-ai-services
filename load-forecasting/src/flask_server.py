import json

import pandas as pd
from flask import Flask, Response, request

from src.model_wrapper import ModelWrapper

expected_content_type = "application/json"
app = Flask(__name__)


def check_header(request):
    return request.content_type == expected_content_type


def read_body(request) -> pd.DataFrame:
    raw = request.get_data()
    data = pd.DataFrame.from_records(json.loads(raw))

    new_idx = pd.to_datetime(data.index.astype(int), unit="ms")
    new_idx = new_idx.tz_localize("utc").tz_convert("Europe/Berlin")

    data.set_index(new_idx, inplace=True, drop=True)
    if pd.infer_freq(data.index) != "h":
        raise ValueError(
            "Incompatible frequency encountered. Expecting hourly resolution."
        )
    else:
        data = data.asfreq("h")

    return data


@app.route("/forecast", methods=["POST"])
def energy_forecast():
    if not check_header(request):
        return Response(
            "Expected content type {} but got {}".format(
                expected_content_type, request.content_type
            ),
            400,
        )

    try:
        history_df = read_body(request)
    except Exception as e:
        print(e)
        return Response("Failed to parse request body", 400)

    try:
        forecast = mw.predict(history_df)
    except Exception as e:
        print(e)
        return Response("Service failed to make a prediction", 500)

    return Response(json.dumps(forecast), 200)


def create_app():
    global mw
    mw = ModelWrapper()
    return app
