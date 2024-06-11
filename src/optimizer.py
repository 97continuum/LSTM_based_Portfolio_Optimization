# -*- coding: utf-8 -*-
"""
functions to run the mean variance portfolio part in the main file

Created on Mon Jun 10
@author: kangj
"""

import numpy as np
import pandas as pd
from datetime import datetime
import scipy.optimize as sc

''' Data Preprocessing ================================================================================ '''
# Rename columns and create date index 
def create_date_index(df, end_Y = 2023, end_M = 12, end_D = 31):
    end_date = datetime(end_Y, end_M, end_D)
    num_periods = len(df)
    date_index = pd.date_range(end=end_date, periods=num_periods, freq='M')
    df.index = date_index
    return df

# Pad the covariance matrices list to a specified length for future use (add zero-matrices at front)
def pad_cov_matrices(cov_matrices, target_length=30):
    num_assets = cov_matrices[0].shape[0]
    padding_length = target_length - len(cov_matrices)
    
    # Create a zero matrix for padding
    zero_matrix = np.zeros((num_assets, num_assets))
    
    # Create a list of zero matrices for padding
    padding = [zero_matrix] * padding_length
    
    # Concatenate the padding and the original list
    padded_cov_matrices = padding + cov_matrices
    
    return padded_cov_matrices

''' MVO optimizer ==================================================================================== '''
# Define portfolio performance functions
def portfolioPerformance(weights, meanReturns, covMatrix):
    """ Calculate Portfolio Performance"""
    annualizedReturns = np.sum(meanReturns * weights) * 252
    annualizedStd = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return annualizedReturns * 100, annualizedStd * 100

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0):
    """ Calculate Negative Sharpe Ratio"""
    annualizedRet, annualizedStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -((annualizedRet - riskFreeRate) / annualizedStd)

def maxSR(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0, 1)):
    """ Maximize the Sharpe Ratio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum up to 1
    bounds = tuple(constraintSet for asset in range(numAssets))
    result = sc.minimize(negativeSR, x0=numAssets * [1. / numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def minimizeVariance(meanReturns, covMatrix, constraintSet=(0, 1)):
    """ Minimize the portfolio variance"""
    numAssets = len(meanReturns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights must sum up to 1
    bounds = tuple(constraintSet for asset in range(numAssets))
    result = sc.minimize(lambda x: portfolioPerformance(x, meanReturns, covMatrix)[1], x0=numAssets * [1. / numAssets], method='SLSQP', bounds=bounds, constraints=constraints)
    return result

''' Weights generator ================================================================================ '''
# Generate dynamic weights based on historical and predicted data
def generate_dynamic_weights(returns, pred_returns, cov_matrices, method, risk_free_rate):
    dates = returns.index
    tickers = returns.columns
    weights = pd.DataFrame(index=dates, columns=tickers).fillna(0)
    
    window = 6  # set rebalance window as 6 months
    for i in range(window, len(dates)-1):  

        # Check if the pred_returns row is NaN
        if pred_returns.iloc[i].isna().any():
            continue
        
        past_data = returns.iloc[i-window:i]
        meanReturns = past_data.mean()
        covMatrix = cov_matrices[i]
        rf = risk_free_rate.iloc[i][0]
        pred_return_next_day = pred_returns.iloc[i+1]
        
        if method == 'max_sr':
            result = maxSR(meanReturns, covMatrix, rf)
        elif method == 'min_var':
            result = minimizeVariance(meanReturns, covMatrix)
        elif method == 'pred_max_sr':
            combined_mean_returns = 0 * meanReturns + 1 * pred_return_next_day  # to be clarified
            result = maxSR(combined_mean_returns, covMatrix, rf)
        elif method == 'pred_min_var':
            combined_mean_returns = 0 * meanReturns + 1 * pred_return_next_day  # to be clarified
            result = minimizeVariance(combined_mean_returns, covMatrix)      

        weights.iloc[i] = result.x
    return weights

''' Backtesting ======================================================================================= '''
# Backtest the portfolio
def backtest(weights, returns, initial_capital=1000):
    portfolio_values = [initial_capital]
    for date in returns.index[1:]:
        prev_value = portfolio_values[-1]
        period_return = np.sum(weights.loc[date] * returns.loc[date])
        portfolio_values.append(prev_value * (1 + period_return))
    return portfolio_values

def backtest_sp500(sp500_data, start_date = '2021-07-31', end_date = '2023-12-31',initial_capital=1000):
    '''Add SP500 monthly data to backtest results'''
    try:
        sp500_data.set_index('Date_prev', inplace=True)
    except: KeyError

    sp500_values = [initial_capital]
    start_idx = sp500_data.index.get_loc(start_date)
    end_idx = sp500_data.index.get_loc(end_date)
    for date in sp500_data.index[start_idx + 1:end_idx+1]:
        prev_value = sp500_values[-1]
        monthly_return = sp500_data.loc[date, 'Return']
        sp500_values.append(prev_value * (1 + monthly_return))
    return sp500_values

''' Portfolio performance metrics ==================================================================== '''
# Calculate cumulative returns
def calculate_cumulative_returns(portfolio_values):
    return (portfolio_values[-1] / portfolio_values[0]) - 1

# Calculate annualized volatility for monthly returns
def calculate_volatility(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return np.std(returns) * np.sqrt(12)  # Annualized volatility for monthly data

# Calculate annualized Sharpe ratio for monthly returns
def calculate_sharpe_ratio(portfolio_values, risk_free_rate):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    risk_free_rate = risk_free_rate.values.flatten()[:len(returns)]  # Ensure risk-free rate series matches returns length
    excess_returns = returns - risk_free_rate  # Risk-free rate is already monthly
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(12)  # Annualized Sharpe ratio for monthly data

# Calculate maximum drawdown
def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def performance_metrics(results, selected_portfolios, risk_free_rate):
    selected_metrics = []
    for name in selected_portfolios:
        values = results[name]
        cumulative_returns = calculate_cumulative_returns(values)
        volatility = calculate_volatility(values)
        sharpe_ratio = calculate_sharpe_ratio(values, risk_free_rate)
        max_drawdown = calculate_max_drawdown(values)
        selected_metrics.append({
            'Portfolio': name,
            'Cumulative Returns': cumulative_returns,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })

    # Create DataFrame for selected metrics
    selected_metrics_df = pd.DataFrame(selected_metrics)
    return selected_metrics_df

''' Weights generator and Backtest with a same start date =============================================== '''
# Generate dynamic weights based on historical and predicted data with a start date
def generate_dynamic_weights_with_start_date(returns, pred_returns, cov_matrices, method, risk_free_rate, start_date):
    dates = returns.index
    tickers = returns.columns
    weights = pd.DataFrame(index=dates, columns=tickers).fillna(0)
    
    window = 6  # set rebalance window as 6 months
    start_date = pd.to_datetime(start_date)  # Convert start_date to Timestamp
    for i in range(window, len(dates)-1):  
        if dates[i] < start_date:
            continue
        past_data = returns.iloc[i-window:i]
        meanReturns = past_data.mean()
        covMatrix = cov_matrices[i]
        rf = risk_free_rate.iloc[i, 0]  # Select the first column's value
        pred_return_next_day = pred_returns.iloc[i+1]
        
        if method == 'max_sr':
            result = maxSR(meanReturns, covMatrix, rf)
        elif method == 'min_var':
            result = minimizeVariance(meanReturns, covMatrix)
        elif method == 'pred_max_sr':
            combined_mean_returns = 0 * meanReturns + 1 * pred_return_next_day  # to be clarified
            result = maxSR(combined_mean_returns, covMatrix, rf)
        elif method == 'pred_min_var':
            combined_mean_returns = 0 * meanReturns + 1 * pred_return_next_day  # to be clarified
            result = minimizeVariance(combined_mean_returns, covMatrix)        

        weights.iloc[i] = result.x
    return weights

# Backtest the portfolio with a same start date
def backtest_with_start_date(weights, returns, initial_capital=1000, start_date='2022-06-30'):
    portfolio_values = [initial_capital]
    start_idx = returns.index.get_loc(start_date)
    for date in returns.index[start_idx + 1:]:
        prev_value = portfolio_values[-1]
        daily_return = np.sum(weights.loc[date] * returns.loc[date])
        portfolio_values.append(prev_value * (1 + daily_return))
    return portfolio_values





