import json
from pathlib import Path

import darts as dts
import numpy as np
import pandas as pd
from darts.models import TFTModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.preprocessing import MinMaxScaler

from src import project_root


class ModelWrapper:
    QUANTILES = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
    FREQ = pd.Timedelta("1h")
    HORIZON = 24

    MODEL_NAME = "DartsTFTAdapter"
    MODEL_ARCH = "_model.pth.tar"
    CHECKPOINT_NAME = "best-epoch=3-val_loss=0.12.ckpt"

    def __init__(self):

        self.model = TFTModel.load(
            str(
                Path(
                    project_root,
                    "models",
                    ModelWrapper.MODEL_NAME,
                    ModelWrapper.MODEL_ARCH,
                )
            ),
            map_location="cpu",
            pl_trainer_kwargs={"accelerator": "gpu"},
        )
        wd = str(Path(project_root, "models"))
        self.model.work_dir = wd
        self.model.load_weights_from_checkpoint(
            model_name=ModelWrapper.MODEL_NAME,
            work_dir=wd,
            file_name=ModelWrapper.CHECKPOINT_NAME,
            map_location="cpu",
        )
        self.model.to_cpu()

    def encode_calendar(self, idx: pd.DatetimeIndex, add_length: int = 0):
        """
        Returns a darts.TimeSeries with the calendar covariates as components
        """
        doy_encoding = datetime_attribute_timeseries(
            idx,
            attribute="dayofyear",
            cyclic=True,
            tz=idx.tz,
            add_length=add_length,
        )

        hod_encoding = datetime_attribute_timeseries(
            idx,
            attribute="hour",
            cyclic=True,
            tz=idx.tz,
            add_length=add_length,
        )
        dow_encoding = datetime_attribute_timeseries(
            idx,
            attribute="dayofweek",
            one_hot=True,
            tz=idx.tz,
            add_length=add_length,
        )
        dts_ts = dts.concatenate([doy_encoding, hod_encoding, dow_encoding], axis=1)
        return dts_ts

    def to_model_format(self, df):
        naive_data_index = pd.DatetimeIndex(
            df.index.tz_convert("utc").tz_localize(None), freq=df.index.freq
        )

        scaler = MinMaxScaler(feature_range=(0, 1))

        np_target = scaler.fit_transform(df.values)
        target = dts.TimeSeries.from_times_and_values(
            naive_data_index,
            np_target,
        )
        known = self.encode_calendar(naive_data_index, ModelWrapper.HORIZON)

        return (target, known), scaler

    def formulate_response(
        self,
        pred: np.ndarray,
        t0: pd.Timestamp,
    ):
        """
        pred: ndarray of shape (ModelWrapper.HORIZON, ModelWrapper.QUANTILES) as is output by the onnx model
        t0: the last time stamp seen in the history

        return: a dict in legacy json format but with quantiles only,
        i.e., without samples
        """

        pred_df = pd.DataFrame(
            data=pred,
            index=pd.date_range(
                start=t0 + ModelWrapper.FREQ,
                periods=ModelWrapper.HORIZON,
                freq=ModelWrapper.FREQ,
            ),
        )

        # restoring legacy nming scheme
        pred_df.columns = [f"power_{q}" for q in ModelWrapper.QUANTILES]

        pred_json_str = pred_df.to_json(date_format="epoch", date_unit="ms")
        pred_dict = {"quantiles": json.loads(pred_json_str)}
        return pred_dict

    def predict(self, data):
        (target, known), scaler = self.to_model_format(data)

        pred = self.model.predict(
            n=ModelWrapper.HORIZON,
            series=target,
            future_covariates=known,
            predict_likelihood_parameters=True,
            verbose=0,
            show_warnings=False,
        )
        assert isinstance(pred, dts.TimeSeries)

        unscaled_pred = scaler.inverse_transform(pred.values())

        response = self.formulate_response(unscaled_pred, t0=data.index[-1])
        return response
