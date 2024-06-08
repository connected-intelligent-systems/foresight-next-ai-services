import json

import pandas as pd
from flask import Flask, request, Response

from src.wu_detection.model_dispatcher import ModelDispatcher

app = Flask(__name__)
md = ModelDispatcher()

expected_content_type = 'application/json'


# restful or not is specified in an argument
@app.route('/estimate', methods=['POST'])
def wake_ups_per_interval():
    if request.content_type == expected_content_type:
        raw = request.get_data(as_text=True)
        try:
            req_dict = json.loads(raw)
            window = pd.DataFrame.from_dict(req_dict, orient="index")
            window.index = pd.to_datetime(window.index)
        except:
            return Response("Failed to parse request body")

        estimate = md.estimate_interval(window)
        return Response(estimate.to_json(), 200)

    else:
        return Response("Expected content type {} but got {}".format(expected_content_type, request.content_type), 400)

def create_app():
    return app

if __name__ == '__main__':
    app.run(debug=True, port=5001, host="0.0.0.0")
