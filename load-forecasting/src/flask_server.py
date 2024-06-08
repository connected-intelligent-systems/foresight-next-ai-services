from pathlib import Path
from typing import Optional
import argparse

import pandas as pd
from flask import Flask, request, Response
from pandas import read_json

from src.model_provider import ModelProvider
from src import project_root


expected_content_type = 'application/json'
app = Flask(__name__)


def check_header(request):
    return request.content_type == expected_content_type


def read_body(request) -> pd.DataFrame:
    raw = request.get_data(as_text=True)
    hist = read_json(raw)
    return hist


@app.route("/forecast", methods=["POST"])
def energy_forecast():
    if not check_header(request):
        return Response("Expected content type {} but got {}".format(expected_content_type, request.content_type), 400)

    try:
        history = read_body(request)
    except Exception as e:
        print(e)
        return Response("Failed to parse request body", 400)

    try:
        forecast = mp.predict(history)
    except Exception as e:
        print(e)
        return Response("Service failed to make a prediction", 500)

    return Response(forecast, 200)


def infer_mounted_model_name(model_path: Path) -> Optional[str]:
    if not model_path.is_dir():
        print('The path of the mounted model is not a directory.')
        return None
    dir_contents = list(model_path.iterdir())
    arch_dirs = [d for d in dir_contents if d.suffix == '']
    if len(dir_contents) < 2 or len(arch_dirs) != 1:
        print('The path of the mounted model must contain exactly one directory describing the architecture'
              'and at least one checkpoint containing the weights.')
        return None
    model_name = arch_dirs[0].parts[-1]
    for d in dir_contents:
        if d.stem != model_name:
            print('Every checkpoint must belong to the same model as the architecture (denoted by file name).')
            return None
    return model_name


def parse_optargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source-model',
                        help='The model to be used for training or serving',
                        type=Path, default=project_root.joinpath('models/global'))
    parser.add_argument('-d', '--dest-model',
                        help='The path where to store the trained model',
                        type=Path, required=False)
    parser.add_argument('-t', '--training-data',
                        help='A path to the file containing the training data',
                        type=Path, required=False)

    return parser.parse_args()


def create_app():
    global mp
    mounted_model_path = project_root.joinpath('models/global')
    mounted_model_name = infer_mounted_model_name(mounted_model_path)
    print(f'Loading model {mounted_model_name}...')
    mp = ModelProvider(mounted_model_path.joinpath(mounted_model_name))
    return app

if __name__ == '__main__':
    args = parse_optargs()

    mounted_model_path = args.source_model
    mounted_model_name = infer_mounted_model_name(mounted_model_path)

    if mounted_model_path.iterdir() and mounted_model_name is not None:
        print(f'Loading model {mounted_model_name}...')
        mp = ModelProvider(mounted_model_path.joinpath(mounted_model_name))
    else:
        print(f'No mounted model detected. Falling back to default global model for training or predictions...')
        mp = ModelProvider()

    if args.dest_model is None and args.training_data is None:
        print('Launching prediction server...')
        app.run(debug=True, port=5001, host="0.0.0.0")
    elif args.dest_model is not None and args.training_data is not None:
        print('Initializing model training...')
        mp.fit(training_data_path=args.training_data, fitted_model_path=args.dest_model)
    else:
        raise ValueError('Whenever dest-model is provided as console argument, training-data is also expected and vice versa.')

