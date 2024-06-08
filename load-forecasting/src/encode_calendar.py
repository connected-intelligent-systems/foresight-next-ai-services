import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def encode_calendar(index: pd.DatetimeIndex,
                    sampling_rate: str = '1H',
                    future_time_steps: int = 0,
                    time_zone: str = 'Europe/Berlin',
                    **kwargs):
    def sin_transformer(period):
        return FunctionTransformer(lambda x: (np.sin(x / period * 2 * np.pi) + 1) / 2)

    def cos_transformer(period):
        return FunctionTransformer(lambda x: (np.cos(x / period * 2 * np.pi) + 1) / 2)

    logging.info('Extending the index...')
    cal_encod = pd.DataFrame(index=index)
    old_index = pd.to_datetime(cal_encod.index, utc=True)
    new_index = pd.date_range(start=old_index[0], end=old_index[-1], freq=sampling_rate).tz_convert(time_zone)
    appendix = pd.date_range(new_index[-1], periods=future_time_steps + 1, freq=sampling_rate)
    cal_encod.index = new_index.union(appendix)

    logging.info('Adding cyclic encoding...')
    cal_encod['sin_hour'] = sin_transformer(24).fit_transform(cal_encod.index.hour)
    cal_encod['cos_hour'] = cos_transformer(24).fit_transform(cal_encod.index.hour)
    cal_encod['sin_day_of_year'] = sin_transformer(366).fit_transform(cal_encod.index.day_of_year)
    cal_encod['cos_day_of_year'] = cos_transformer(366).fit_transform(cal_encod.index.day_of_year)

    logging.info('Adding categorical encoding...')
    day_codes = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    cal_encod['day_of_week'] = pd.Categorical(cal_encod.index.day_of_week, categories=list(range(7)))
    oh_days = pd.get_dummies(cal_encod['day_of_week']).astype('category')
    oh_days.columns = day_codes
    cal_encod = pd.concat([cal_encod, oh_days], axis=1)
    cal_encod.drop(['day_of_week'], axis=1, inplace=True)

    cal_encod.index = cal_encod.index.tz_convert('UTC')

    return cal_encod
