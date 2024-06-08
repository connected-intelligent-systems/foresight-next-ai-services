import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

from src.NILM_Dataset import NILMDataset
from src.utils import make_windows_from_list_of_chunks, remove_smaller_chunks_and_pad_noise
from src.utils import mark_small_gaps, chunkize_df_by_nans


def preprocess(df, settings):
    X_scaler = settings['X_scaler']

    df.columns = ['power']
    df['power'] = pd.to_numeric(df['power'])

    df['power'] = X_scaler.transform(df['power'].values.reshape(-1, 1)).squeeze()

    df.index = pd.to_datetime(df.index)
    df = df.groupby(df.index).mean()
    df = df.resample(settings['SAMPLING_RATE']).mean()
    df.ffill(limit=10, inplace=True)

    df['unix'] = df.index.astype('int64') // 10 ** 9

    test = NILMDataset(x=df['power'],
                       y=df['power'],
                       status=df['unix'],
                       window_size=settings['WINDOW_SIZE'],
                       stride=settings['WINDOW_SIZE']  # non-overlapping windows
                       )
    test_loader = data_utils.DataLoader(test, batch_size=64,
                                        shuffle=False,
                                        pin_memory=False
                                        )

    return test_loader


def preprocess_old(df, settings):
    SAMPLING_RATE = settings['SAMPLING_RATE']

    SMALL_GAP_LIMIT = settings['SMALL_GAP_LIMIT']
    SMALL_FILL_WITH = settings['SMALL_FILL_WITH']
    WINDOW_SIZE = settings['WINDOW_SIZE']
    X_scaler = settings['X_scaler']

    df.columns = ['power']
    df['power'] = pd.to_numeric(df['power'])

    # orig_tz = pd.to_datetime(df.index).tz

    df.index = pd.to_datetime(df.index)
    df = df.groupby(df.index).mean()
    df = df.resample(SAMPLING_RATE).mean()
    df.fillna(method='ffill', limit=1, inplace=True)

    if df.isna().sum()['power'] > 0:
        small_gaps = mark_small_gaps(df['power'], limit=SMALL_GAP_LIMIT)
    else:
        small_gaps = np.zeros_like(df).squeeze()
        small_gaps = ~(small_gaps == 0)

    df.rename(columns={'power': 'sm_power'}, inplace=True)
    df['app_power'] = np.nan

    df['sm_power'] = X_scaler.transform(df['sm_power'].values.reshape(-1, 1)).squeeze()

    df.loc[small_gaps, 'sm_power'] = SMALL_FILL_WITH

    chunks = chunkize_df_by_nans(df)
    # %%
    chunks = remove_smaller_chunks_and_pad_noise(chunks_lst=chunks, window_size=WINDOW_SIZE,
                                                 noise_value=SMALL_FILL_WITH, _sampling_rate=SAMPLING_RATE)

    timesteps, X, _ = make_windows_from_list_of_chunks(chunks, window_size=WINDOW_SIZE)
    # app.logger.info(orig_tz)
    return timesteps, X


def _predict(_test_loader, settings):
    _model = settings['model']
    scaler_apl = settings['y_scaler']

    ts_list = []
    logits_ys = []
    _model.eval()
    with torch.no_grad():
        for batch in _test_loader:
            x, _, ts = [batch[i] for i in range(3)]

            x = x.float()
            logits = _model(x)

            ts_list.append(ts.numpy().squeeze())
            logits_ys.append(logits.numpy().squeeze())

    ts_list = np.concatenate(ts_list)
    y_pred = np.concatenate(logits_ys)

    ts_list = ts_list.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # removing elements which were added to make window size standard
    idx = np.where(ts_list > 1)
    ts_list = ts_list[idx]
    y_pred = y_pred[idx]

    y_pred = scaler_apl.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    y_pred = np.maximum(y_pred, 0)

    return_df = pd.DataFrame(data={'preds': y_pred, 'unix': ts_list})
    return_df['time'] = pd.to_datetime(return_df['unix'], unit='s')
    return_df.set_index('time', inplace=True, drop=True)
    del return_df['unix']

    return return_df


def _predict_old(_X, _timesteps, settings):
    model = settings['model']
    y_scaler = settings['y_scaler']

    preds = model.predict(_X)
    preds = y_scaler.inverse_transform(preds).squeeze()

    return_df = pd.DataFrame(data={'preds': preds}, index=_timesteps)
    return return_df
