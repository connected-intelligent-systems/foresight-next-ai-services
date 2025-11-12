from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, request, Response
from pandas import read_json

from src.model_provider import ModelProvider
from src import project_root


expected_content_type = "application/json"
app = Flask(__name__)


def check_header(request):
    return request.content_type == expected_content_type


def parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ["t", "true"]:
        return True
    elif s in ["f", "false"]:
        return False
    else:
        raise ValueError(f"Failed to parse string parameter {s}.")


def read_params(request):
    n_samples = request.args.get("n_samples")
    return_samples = request.args.get("return_samples")
    n_samples = 100 if n_samples is None else int(n_samples)
    return_samples = True if return_samples is None else parse_bool(return_samples)
    return n_samples, return_samples


def read_body(request) -> pd.DataFrame:
    raw = request.get_data(as_text=True)
    hist = read_json(raw)
    return hist


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
        n_samples, return_samples = read_params(request)
    except Exception as e:
        print(e)
        return Response("Failed to parse query parameters", 400)

    try:
        history = read_body(request)
    except Exception as e:
        print(e)
        return Response("Failed to parse request body", 400)

    try:
        forecast = mp.predict(
            history,
            n_samples=n_samples,
            return_samples=return_samples,
        )
    except Exception as e:
        print(e)
        return Response("Service failed to make a prediction", 500)

    return Response(forecast, 200)


def create_app():
    global mp
    mounted_model_path = project_root.joinpath("models/global")
    mounted_model_name = "global"
    print(f"Loading model {mounted_model_name}...")
    mp = ModelProvider(
        mounted_model_path.joinpath(mounted_model_name),
        # n_samples=args.n_samples,
    )
    return app
