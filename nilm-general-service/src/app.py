import math
import os

import torch
from flask import Flask, make_response, request

from src.Electricity_model_NILMTK import NILMTKModel
# from ml_utils import preprocess, _predict
from src.ml_utils import preprocess, _predict
from src.utils import json2df
from src.utils import ld_pkl
from src.utils import read_json, df2json

app = Flask(__name__)

# SAMPLING_RATE = None
# SMALL_GAP_LIMIT = None
# SMALL_FILL_WITH = None
orig_tz = None

# X_scaler, y_scaler = None, None
global_dict = {}


# @app.before_first_request
def init():
    global global_dict

    # global_dict['WINDOW_SIZE'] = 480

    appliance_name = os.environ['APPLIANCE']  # ["DW", "FRDFZ", "KTL", "MW", "TV" "WM"] # , "SM"
    assert appliance_name in ["DW", "FRDFZ", "KTL", "MW", "TV", "WM"]
    house_id = int(os.environ['hid'])
    assert house_id in [5, 9]

    acronym2full = {"DW": 'Dishwasher', "FRDFZ": 'Fridge-Freezer', "KTL": 'Kettle', "MW": 'Microwave', "TV": 'TV',
                    "WM": 'Washing_Machine'}
    full_apl_name = acronym2full[appliance_name]

    model_path = os.path.join(f'./models/{house_id}_{full_apl_name}', 'best_acc_model.pth')
    model_dict = torch.load(model_path, map_location='cpu')

    scaler_SM = ld_pkl('./scalers/SM.pkl')
    scaler_apl = ld_pkl(f'./scalers/{appliance_name}.pkl')

    print(f'\n\t\tModel loaded: {model_path}')
    print(f'\t\tSM Scaler loaded: ./scalers/SM.pkl')
    print(f'\t\tAPL Scaler loaded: ./scalers/{appliance_name}.pkl')
    print('\n')

    SETTINGS_JSON = 'settings.json'
    settings_dict = read_json(SETTINGS_JSON)
    #
    WINDOW_SIZE = settings_dict['general']['WINDOW_SIZE']
    SAMPLING_RATE_N = settings_dict['general']['SAMPLING_RATE_N']
    WINDOW_SIZE_MINUTES = WINDOW_SIZE * SAMPLING_RATE_N / 60
    SMALL_GAP_SECONDS = settings_dict['general']['SMALL_GAP_SECONDS']
    SMALL_FILL_WITH = settings_dict['general']['SMALL_FILL_WITH']

    SAMPLING_RATE = f'{SAMPLING_RATE_N}s'
    SMALL_GAP_LIMIT = math.ceil(SMALL_GAP_SECONDS / SAMPLING_RATE_N)

    model = NILMTKModel(window_size=WINDOW_SIZE, drop_out=0.1)
    model.load_state_dict(model_dict)
    global_dict['SAMPLING_RATE'] = SAMPLING_RATE
    global_dict['SMALL_GAP_LIMIT'] = SMALL_GAP_LIMIT
    global_dict['SMALL_FILL_WITH'] = SMALL_FILL_WITH
    global_dict['WINDOW_SIZE'] = WINDOW_SIZE
    global_dict['WINDOW_SIZE_MINUTES'] = WINDOW_SIZE_MINUTES
    global_dict['X_scaler'] = scaler_SM
    global_dict['y_scaler'] = scaler_apl
    global_dict['model'] = model


@app.route("/")
def usage_guide():
    return "Please use /predict with POST method and the format of the data should be as follows: ..."


def handle_input(_json):
    global global_dict
    df = json2df(_json)  # Datetimeindex=time, column = power
    test_loader = preprocess(df, global_dict)

    try:
        return_df = _predict(_test_loader=test_loader, settings=global_dict)

        return df2json(return_df)
    except Exception as e:
        print(f"in handle_input() {e}")
        return None


@app.route("/predict", methods=["POST"])
def predict():
    global global_dict

    app.logger.info(f'\tPredicting')
    if request.is_json:
        req_json = request.get_json()

        response = handle_input(req_json)
        if response is None:
            app.logger.info('No json')
            return f"Couldn't make enough windows out of data. Send atleast a bit more than {global_dict['WINDOW_SIZE'] + 1} samples of data.", 400

        res = make_response(response, 200)
        res.headers['Content-Type'] = "application/json; charset=utf-8"

        return res
    else:
        app.logger.info('No json')
        return "No JSON received", 400

def create_app():
    init()
    return app

if __name__ == '__main__':
    init()
    app.run(debug=True, host="0.0.0.0")
