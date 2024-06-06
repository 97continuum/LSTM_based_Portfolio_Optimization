# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 23:51:41 2024

@author: 13370
"""


import numpy as np
import pandas as pd

def calculate_technical_indicators(df):
    df['return_momentum_3m'] = df['trt1m'].rolling(window=3).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    df['high_low_ratio'] = df['prchm'] / df['prclm']
    df['RSI_14'] = df['prccm'].rolling(window=15).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0)).sum()))))
    df['MA_3'] = df['prccm'].rolling(window=3).mean()
    df['price_to_MA_3'] = df['prccm'] / df['MA_3']
    df['return_momentum_6m'] = df['trt1m'].rolling(window=6).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    df['MA_6'] = df['prccm'].rolling(window=6).mean()
    df['return_momentum_9m'] = df['trt1m'].rolling(window=9).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    df['MA_9'] = df['prccm'].rolling(window=9).mean()
    df['return_momentum_12m'] = df['trt1m'].rolling(window=12).apply(lambda x: np.prod(1 + x) - 1, raw=False)
    df['MA_12'] = df['prccm'].rolling(window=12).mean()
    return df