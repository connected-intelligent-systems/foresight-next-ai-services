import json
import math
import pickle

import numpy as np
import pandas as pd
# import tensorflow as tf


def read_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def ld_pkl(path):
    return pickle.load(open(path, 'rb'))


def json2df(py_dict):
    def form_lists(arr_objs_json):
        if isinstance(arr_objs_json, str):
            arr_objs_json = json.loads(arr_objs_json)
        indices = []
        values = []
        for obj in arr_objs_json:
            indices.append(pd.to_datetime(obj['time']))
            values.append(obj['value'])
        return indices, values

    _indices, _values = form_lists(py_dict)
    df = pd.DataFrame(data=_values, index=_indices)

    df.index.name = 'time'
    # df.rename(columns={0: 'power'}, inplace=True)

    return df


def df2json(df):
    arr_objs = []
    i = 0
    for ix in df.index:
        arr_objs.append({"time": (ix.isoformat()), "value": str(df.loc[ix].values[0])})
    py_json = json.dumps([ob for ob in arr_objs])

    return py_json


def load_model(model_path):
    """ Loads a model from a specified location.
    Parameters:
    model (tensorflow.keras.Model): The Keas model to which the loaded weights will be applied to.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.
    """

    # model_name = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"

    print("PATH NAME: ", model_path)

    model = tf.keras.models.load_model(model_path)
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model


def convert2unix(ts: pd.Timestamp):
    """
    Converts pandas timestamp to UNIX timestamp
    :param ts:
    :return:
    """
    return (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')


def convert2timestamp(ux: int):
    """
    Converts UNIX timestamp to pandas timestamp
    :param ux:
    :return:
    """
    # return pd.Timestamp(datetime.fromtimestamp(ux, tz=None))
    return pd.to_datetime(ux, unit='s')


def mark_small_gaps(tser: pd.Series, limit: int):
    """
    Fills smaller gaps i.e. consecutive Nans upto `limit` with the `fill_value`
    :param tser:
    :param limit:
    :param fill_value:
    :return:
    """

    def count_len(a):
        return len(a)

    _mask = np.isnan(tser)  # finds Nans in the series
    _df = pd.DataFrame((tser, _mask), index=['value', 'mask']).transpose()
    _df['mask'] = _mask
    _df['inversed_mask'] = ~_mask
    _df['inversed_mask_cumsum'] = (~_mask).cumsum()  # Count if there is no Nan i.e. stops counting when Nan occurs

    _df.value = _df['value'].astype(float)

    # group values where there is Nan in the series by "stopped counting"
    _groups = _df[np.isnan(_df['value'])].groupby('inversed_mask_cumsum')

    # count those nans
    _df['count'] = _df['inversed_mask_cumsum'].map(_groups.apply(count_len))

    # reset counting where there are no nans
    _df.loc[~np.isnan(_df['value']), 'count'] = 0
    _df['value2'] = _df['value'].copy(deep=True)

    # _df.loc[(_df['count'] <= limit) & (_df['count'] > 0), 'value2'] = fill_value

    return (_df['count'] <= limit) & (_df['count'] > 0)


def chunkize_df_by_nans(_df: pd.DataFrame, col_name='sm_power'):
    def get_start_end_block(bl: pd.Series):
        return bl.index[0], bl.index[-1]

    def split_by_nans(ser: pd.Series):
        # https://stackoverflow.com/questions/21402384/how-to-split-a-pandas-time-series-by-nan-values/66015224#66015224
        sparse_ts = ser.astype(pd.SparseDtype('float'))
        block_locs = zip(
            sparse_ts.values.sp_index.to_block_index().blocs,
            sparse_ts.values.sp_index.to_block_index().blengths,
        )
        blocks = [
            ser.iloc[start: (start + length - 1)]
            for (start, length) in block_locs
        ]
        return blocks

    _tmp_ser = _df[col_name].copy(deep=True)
    _blocks = split_by_nans(_tmp_ser)

    _chunks = []
    for idx, _bl in enumerate(_blocks):
        if len(_bl) > 0:  # only considering chunks with at least one sample
            bl_st, bl_ed = get_start_end_block(_bl)
            _chunk = _df[bl_st:bl_ed].copy(deep=True)
            _chunks.append(_chunk)
    return _chunks


def read_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def edit_json(file, appliance, parameter, parameter_value):
    data = read_json(file)
    data[appliance][parameter] = parameter_value
    with open(file, 'w') as fp:
        json.dump(data, fp, indent=4, sort_keys=True)


def remove_smaller_chunks_and_pad_noise(chunks_lst, window_size, noise_value, _sampling_rate):
    """
    Consider only chunks with atleast one window and pad them with noise values.
    :param _sampling_rate:
    :param noise_value:
    :param window_size:
    :param chunks_lst:
    :return:
    """

    def pad_noise(chunk: pd.DataFrame, _window_size: int, sampling_rate: int, fill_value: int):
        cols = chunk.columns
        pad_size = math.floor(_window_size / 2)
        st_idx = pd.date_range(start=chunk.index[0] - pad_size * pd.Timedelta(sampling_rate),
                               freq=f'{sampling_rate}', periods=pad_size)
        st_df = pd.DataFrame(data={cols[0]: [fill_value] * pad_size, cols[1]: [fill_value] * pad_size}, index=st_idx)

        ed_idx = pd.date_range(end=chunk.index[-1] + pad_size * pd.Timedelta(sampling_rate),
                               freq=f'{sampling_rate}', periods=pad_size)
        ed_df = pd.DataFrame(data={cols[0]: [fill_value] * pad_size, cols[1]: [fill_value] * pad_size}, index=ed_idx)

        st_df.index.name = 'time'
        ed_df.index.name = 'time'

        return pd.concat([st_df, chunk, ed_df])

    new_chunks_lst = []
    for _chunk in chunks_lst:
        if len(_chunk) >= window_size:
            _chunk_ = pad_noise(chunk=_chunk, _window_size=window_size, sampling_rate=_sampling_rate,
                                fill_value=noise_value)
            new_chunks_lst.append(_chunk_)
    return new_chunks_lst


# Code for making windows

def get_center_value(window):
    idx = math.floor(len(window) / 2)
    return window[idx]


def get_number_of_windows(data, kernel_size, padding=0, stride=1):
    return len(data) - kernel_size - 2 * padding / stride + 1


def create_windows(iterable, window_size):
    i = iter(iterable)
    win = []
    for e in range(0, window_size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win


def get_windows(df, window_size=5):
    w = []
    for col in df.columns:
        w.append(list(create_windows(df[col], window_size)))
    return w


def make_windows_of_chunk(trainset, window_size):
    _main = []
    _center = []
    _ts = []

    # columns = trainset.columns
    windows_set = get_windows(trainset, window_size)

    for window_idx in range(len(windows_set[0])):
        _ts.append(windows_set[0][window_idx][0])
        _main.append(windows_set[1][window_idx])
        _center.append(get_center_value(windows_set[2][window_idx]))

    # print(f"Number of windows {get_number_of_windows(trainset[columns[0]], kernel_size=window_size, padding=0, stride=1)}")
    # list_tuples = list(zip(ts, main, center))
    return _ts, _main, _center  # pd.DataFrame(list_tuples, columns=columns)


def make_windows_from_list_of_chunks(chunks_lst, window_size):
    compl_ts, compl_power_windows, compl_app_scalar = [], [], []
    for _chunk in chunks_lst:
        # if len(_chunk) > window_size:
        lst_ts, lst_power_windows, lst_app_scalar = make_windows_of_chunk(_chunk.reset_index(),
                                                                          window_size=window_size)
        compl_ts += lst_ts
        compl_power_windows += lst_power_windows
        compl_app_scalar += lst_app_scalar

    return np.array(compl_ts), np.array(compl_power_windows), np.array(compl_app_scalar)
