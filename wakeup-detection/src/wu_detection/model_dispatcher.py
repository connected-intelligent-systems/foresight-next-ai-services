from src.wu_detection.estimator import Estimator
from src.wu_detection.preprocessor import Preprocessor
from src.wu_detection.wake_up import WakeUp


class ModelDispatcher:

    def __init__(self):
        self.estimator = Estimator()
        self.window_size = self.estimator.stage1.input.shape[1]
        self.n_features = self.estimator.stage1.input.shape[2]
        self.pp = Preprocessor(self.window_size)

    # window must be a datetime-indexed data frame with one power column
    def estimate_interval(self, window):
        energy_data, window_start = self.pp.prepare(window)

        print(energy_data.shape)

        if energy_data is None:
            raise AssertionError("No data provided.")
        elif energy_data.shape != (self.window_size,):
            raise AssertionError("Ill-shaped data provided")
        else:

            s1_conf = self.estimator.stage1_estimate(energy_data)
            wu_time = self.estimator.stage2_estimate(energy_data, window_start)
            return WakeUp(s1_conf[0], s1_conf[1], s1_conf[2], wu_time)
