def scale_resample(data, freq, scaling):
    resampled = data.resample(freq).mean()
    resampled.iloc[:, 0] = resampled.iloc[:, 0] / scaling
    interpolated = resampled.interpolate()
    return interpolated


class Preprocessor:
    def __init__(self, window_size, freq='T', scaling=4000):
        self.window_size = window_size
        self.freq = freq
        self.scaling = scaling

    # window must be a datetime-indexed data frame with one power column
    def prepare(self, window):
        if window.empty:
            raise AssertionError("Not enough data to interpolate")

        resampled = scale_resample(window, freq=self.freq, scaling=self.scaling)

        if len(resampled) < self.window_size:
            raise AssertionError("Not enough data to interpolate")
        elif len(resampled) > self.window_size:
            resampled = resampled.tail(self.window_size)

        power = resampled.values.flatten()
        start_time = resampled.index[0]

        return power, start_time
