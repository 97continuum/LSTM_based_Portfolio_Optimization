# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 23:51:41 2024

@author: 13370
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_stock_data(stock):
    # Data typr and unit processing
    stock['trt1m'] = stock['trt1m'] / 100
    stock['datadate'] = pd.to_datetime(stock['datadate'])

    # Summary of stocks
    stock_summary = stock.groupby('tic').size().reset_index(name='number')
    stock_summary = stock_summary[stock_summary['number'] == 168]

    
    # Stocks that experienced a consecutive three-day decline in the last seven days are ruled out
    stock_1 = stock[stock['tic'].isin(stock_summary['tic'])].copy()
    stock_1 = (stock_1.sort_values(['tic', 'datadate'])
                        .groupby('tic')
                        .apply(lambda x: x.tail(9))
                        .reset_index(drop=True))
    stock_1['consecutive_neg'] = (stock_1.groupby('tic')['trt1m']
                                 .transform(lambda x: x.shift(2) < 0) &
                                 stock_1.groupby('tic')['trt1m']
                                 .transform(lambda x: x.shift(1) < 0) &
                                 (stock_1['trt1m'] < 0))
    stock_summary_exclude = (stock_1.groupby('tic')
                         .agg(exclude=('consecutive_neg', 'any'))
                         .reset_index())
    stock_1_tics = stock_summary_exclude[~stock_summary_exclude['exclude']]['tic']
    
    
    # Calculate average returns
    stock_2 = stock[stock['tic'].isin(stock_1_tics)].copy()
    stock_2.dropna(subset=['trt1m'], inplace=True)
    stock_2 = stock_2.groupby('tic').head(135)
    stock_2['avg'] = stock_2.groupby('tic')['trt1m'].transform('mean')
    stock_2 = stock_2.groupby('tic').tail(1).reset_index(drop=True)
    stock_2.sort_values(by='avg', ascending=False, inplace=True)

    # Select stocks tickers based on average returns
    avg_base_select = []
    count = np.zeros(100)
    gic_all = stock_2['gsector'].unique()

    for i in range(277):
        gic_index = stock_2.iloc[i]['gsector']
        count[gic_index] += 1
        if count[gic_index] > 5:
            continue
        avg_base_select.append(stock_2.iloc[i]['tic'])
        if all(count[gic_all] >= 2):
            break

    # Select stocks
    stock_avg = stock[stock['tic'].isin(avg_base_select)].dropna()
    stock_avg['yymm'] = stock_avg['datadate'].dt.to_period('M')
    
    return stock_avg


def clean_financial_ratios(stock_avg, stock_factor):
    stock_factor_1 = stock_factor.drop(columns=[stock_factor.columns[2], stock_factor.columns[3]])
    stock_factor_1 = stock_factor_1[stock_factor_1['gvkey'].isin(stock_avg['gvkey'])]
    stock_factor_1['yymm'] = pd.to_datetime(stock_factor_1['public_date']).dt.to_period('M')
    return stock_factor_1

def merge_and_clean_data(stock_avg, stock_factor_1):
    col_to_fill = stock_factor_1.columns[3:70]
    stock_all = pd.merge(stock_avg, stock_factor_1, on=['yymm', 'gvkey'], how='outer')
    stock_all.sort_values(by=['gvkey', 'yymm'], inplace=True)
    stock_all[col_to_fill] = stock_all.groupby('gvkey')[col_to_fill].fillna(method='ffill')
    stock_all_final = stock_all[(stock_all['yymm'] >= '2010-01') & stock_all['tic'].notna() & ~stock_all['tic'].isin(['BX', 'AVGO'])]
    return stock_all_final

def filter_useful_features(stock_all_final):
    na_counts = stock_all_final.isna().sum()
    stock_final = stock_all_final.loc[:, na_counts == 0]
    return stock_final

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

def merge_macro_data(stock_use, macro):
    macro['yymm'] = pd.to_datetime(macro['caldt']).dt.to_period('M')
    stock_use = stock_use.merge(macro, on='yymm', how='left')
    stock_use.drop(columns=['caldt','yymm'], inplace=True)
    return stock_use

def normalize_data(stock_use):
    stock_n = stock_use.copy()
    normalizer = MinMaxScaler()
    stock_n.iloc[:, 6:79] = normalizer.fit_transform(stock_n.iloc[:, 6:79])
    return stock_n