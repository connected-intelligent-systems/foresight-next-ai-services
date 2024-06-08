import json
from pathlib import Path
from typing import Union, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.utils.missing_values import extract_subseries

from src import project_root
from src.encode_calendar import encode_calendar
from src.scale import fit_scaler, scale_data


class ModelProvider:
    def __init__(
        self,
        model_path: Union[Path, str] = project_root.joinpath('models/global/global')
    ):
        dummy_index = pd.date_range(start=pd.Timestamp(2000, 1, 1), freq='1h', periods=8 * 24)
        dummy_data = pd.DataFrame(index=dummy_index, columns=['power'], data=np.zeros(shape=(8 * 24)))
        dummy_series = TimeSeries.from_dataframe(dummy_data)
        dummy_covar = TimeSeries.from_dataframe(encode_calendar(index=dummy_index, future_time_steps=25))

        self.model_path = model_path
        self.model = self.recreate_model(target_series=dummy_series, covariates=dummy_covar)

    def fit(
        self,
        training_data_path: Union[Path, str],
        fitted_model_path: Union[Path, str]
    ):
        # Start by creating the parent folder so that path and permission issues become evident before training
        fitted_model_path = fitted_model_path.joinpath(fitted_model_path.parts[-1])
        fitted_model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Parent directory {fitted_model_path.parent} created')

        data = pd.read_csv(training_data_path, index_col=0, parse_dates=True)
        scaler, scaled_ts, future_covar = _encode_data(data)

        subseries = extract_subseries(scaled_ts)
        subseries_covar = [future_covar] * len(subseries)

        self.model.fit(series=subseries, future_covariates=subseries_covar, epochs=1, num_loader_workers=4)
        print('Model trained.')

        self.model.save(str(fitted_model_path))
        print(f'Model saved at {fitted_model_path}.')

    def predict(
        self,
        historic_data: pd.DataFrame,
        horizon:int=24,
        n_samples:int=100) -> str:
        scaler, scaled_ts, future_covar = _encode_data(historic_data)

        pred = self.model.predict(horizon, series=scaled_ts, future_covariates=future_covar, num_samples=n_samples)
        scaled_pred_ts = scale_data([pred], scaler, inverse_transform=True)[0]
        scaled_pred = scaled_pred_ts.pd_dataframe().to_json()
        quant_pred = scaled_pred_ts.quantiles_df(quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)).to_json()

        composite_pred = dict()
        composite_pred['samples'] = json.loads(scaled_pred)
        composite_pred['quantiles'] = json.loads(quant_pred)

        return json.dumps(composite_pred)

    # Workaround for the bug causing some tensors to be loaded on the wrong device in probabilistic models
    def recreate_model(
        self,
        target_series: Union[TimeSeries, Iterable[TimeSeries]],
        covariates: Union[TimeSeries, Iterable[TimeSeries]],
        new_model_path: Optional[Union[str, Path]] = None,
        reset_weights: bool = False
    ) -> TorchForecastingModel:
        model_path = new_model_path or self.model_path
        model = TorchForecastingModel.load(str(model_path), map_location='cpu')
        _purge_model_trainer(model)
        recreated_model = model.untrained_model()

        if not reset_weights and model.model is not None:
            recreated_model.fit(target_series,
                                future_covariates=covariates,
                                val_series=target_series,
                                val_future_covariates=covariates,
                                epochs=1,
                                max_samples_per_ts=1)
            recreated_model.model.load_state_dict(model.model.state_dict())

        return recreated_model


def _purge_model_trainer(model: TorchForecastingModel):
    model.trainer_params['logger'] = []
    model.trainer_params['callbacks'] = []
    model.trainer_params['lr_scheduler_cls'] = None
    model.trainer_params['lr_scheduler_kwargs'] = None
    model.model_params['pl_trainer_kwargs']['logger'] = []
    model.model_params['pl_trainer_kwargs']['callbacks'] = []
    model.model_params['pl_trainer_kwargs']['accelerator'] = 'auto'
    model.model_params['pl_trainer_kwargs']['devices'] = 1
    model.model_params['lr_scheduler_cls'] = None
    model.model_params['lr_scheduler_kwargs'] = None
    model.model_params['save_checkpoints'] = False

def _encode_data(data: pd.DataFrame, future_time_steps=25, lower_bound_zero=True) -> Tuple[Scaler, TimeSeries, TimeSeries]:
    ts = TimeSeries.from_dataframe(data)

    if lower_bound_zero:
        scaler = fit_scaler(target_series=[ts.append_values(np.zeros(shape=(1, 1, 1)))])
    else:
        scaler = fit_scaler(target_series=[ts])

    scaled_ts = scale_data([ts], scaler)[0]
    future_covar = TimeSeries.from_dataframe(encode_calendar(index=ts.time_index, future_time_steps=future_time_steps))
    return scaler, scaled_ts, future_covar


if __name__ == '__main__':
    data_path = project_root.joinpath('sample_data/ccef3bd47b7d8a93578791579362d7f61e6daed3dd4acbab47997bc8927218f.csv')
    fitted_model_path = project_root.joinpath('tmp/models/tuned_global')

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    json_str = data.head(70 * 24).to_json()
    parsed_json_data = pd.read_json(json_str)

    mp = ModelProvider()
    #print(mp.model.__dict__)

    mp.fit(training_data_path=data_path, fitted_model_path=fitted_model_path)
    pred = mp.predict(historic_data=parsed_json_data)

    print(pred)
