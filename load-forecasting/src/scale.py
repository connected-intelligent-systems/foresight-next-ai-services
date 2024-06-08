from typing import Sequence

import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from matplotlib import pyplot as plt


def scaling_successful(scaled_series: Sequence[TimeSeries], plot: bool = False) -> bool:
    successes = [0 <= np.min(s.values()) <= np.max(s.values()) <= 1 for s in scaled_series]
    if plot:
        for s in scaled_series:
            s.plot()
        plt.show()
    return all(successes)


def fit_scaler(target_series: Sequence[TimeSeries],
               fit_step_limit: int = 0) -> Scaler:
    if fit_step_limit > 0:
        target_series = [ts[:fit_step_limit] for ts in target_series]

    mm_scaler = Scaler()
    mm_scaler.fit(target_series)

    return mm_scaler


def scale_data(target_series: Sequence[TimeSeries],
               fitted_scaler: Scaler,
               inverse_transform: bool = False,
               safety_margin: float = 0.2,
               **kwargs) -> Sequence[TimeSeries]:
    if inverse_transform:
        scaled_data = fitted_scaler.inverse_transform(target_series)
        scaled_data = [s * (1 + safety_margin) for s in scaled_data]
    else:
        unscaled = [s / (1 + safety_margin) for s in target_series]
        scaled_data = fitted_scaler.transform(unscaled)

    return scaled_data
