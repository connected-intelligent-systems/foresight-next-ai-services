from datetime import timedelta

import numpy
from pkg_resources import resource_filename
from tensorflow.keras.models import load_model


class Estimator:

    def __init__(self):
        stage1_file = resource_filename('res', 'wu_stage1_cnn.h5')
        stage2_file = resource_filename('res', 'wu_stage2_gru.h5')
        self.stage1 = load_model(stage1_file, compile=False)
        self.stage2 = load_model(stage2_file, compile=False)
        self.stage1.compile(loss='categorical_crossentropy')
        self.stage2.compile(loss='mse')

    def stage1_estimate(self, data):
        data = data.reshape(1, -1, 1)
        verdict = (self.stage1.predict(data)).flatten()
        return verdict

    def stage2_estimate(self, data, window_start):
        data = data.reshape(1, -1, 1)
        wu_ratio = (self.stage2.predict([data])).flatten()
        wu_ratio = 0 if numpy.isnan(wu_ratio) else wu_ratio
        wu_offset = int(data.shape[1] * wu_ratio)
        wu_estimate = window_start + timedelta(minutes=wu_offset)
        return wu_estimate
