""" Helper Script to hold functions to run main file
"""

import numpy as np
import pandas as pd
import os
from lstm_class import Lstm
from sklearn.metrics import mean_squared_error


def create_df_per_stock(tickers:list, dataframe:pd.DataFrame) -> dict:
    """ Create a DataFrame for each individual stock and store it in a Dictionary
        
        Shift the Target Return up because we want to use X features at t to predict returns at t+1
    Args:
        list_of_stocks (list): List of Unique Stock Tickers
        dataframe (pd.DataFrame): DataFrame of Stock Features and Target Returns

    Returns:
        dict: Dictionary with key (stock ticker) and value (dataframe of features and returns)
    """
    df_per_stock = {}
    for stock in tickers:
        df = dataframe[dataframe['tic'] == stock]
        df['trt1m'] = df['trt1m'].shift(-1) # Shift Target Return up
        df = df.dropna() # Drop any rows with NA values
        df_per_stock[stock] = df
    return df_per_stock

def prepare_data(df:pd.DataFrame, sequence_length:int = 12) -> tuple:
    """ Split DataFrame into X Features and y target returns

    Args:
        df (pd.DataFrame): DataFrame for the Stock with Features and Target
        sequence_length (int, optional): Number of Months for Time Sequence. Defaults to 12.

    Returns:
        tuple: X and y numpy arrays representing the Features and Target Returns respectively
    """
    y = df['trt1m'].values
    df.drop(columns=['trt1m'], inplace=True)
    X = df.iloc[:, 2:].values
    X_features, y_target = [], []
    for i in range(X.shape[0] - sequence_length):
        X_features.append(X[i:i+sequence_length])
        y_target.append(y[i + sequence_length])
    X_features = np.array(X_features)
    y_target = np.array(y_target)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X_features[:train_size], y_target[:train_size]
    X_test, y_test = X_features[train_size:], y_target[train_size:]
    return X_train, y_train, X_test, y_test

def check_existing_results(ticker, folder='../results'):
    """ Check if the HyperParameter Tuning Results CSV File exists for current Ticker

    Args:
        ticker (_type_): Ticker
        results_folder (str, optional): Results Folder. Defaults to 'results'.

    Returns:
        _type_: _description_
    """
    filename = os.path.join(folder, f"{ticker}_hyperparameter_tuning_results.csv")
    return os.path.exists(filename), filename

def run_hyperparameter_tuning(X_train:np.array,
                              y_train:np.array,
                              param_grid:dict,
                              csv_path) -> dict:
    """ Run Hyperparameter Tuning and return a dictionary of best configuration

    Args:
        X_train (_type_): _description_
        y_train (_type_): _description_
        param_grid (_type_): _description_
        csv_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    lstm_model = Lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    best_config = lstm_model.hyperparameter_tuning(X_train,
                                                   y_train,
                                                   param_grid,
                                                   n_splits=5,
                                                   csv_path=csv_path)
    return best_config

def train_and_evaluate_best_model(all_models, ticker, X_train, y_train, X_test, y_test, best_config):
    """ Train and Evaluate the Model with the Best Configuration for Each Ticker
        Predict on the Testing Set and 

    Args:
        ticker (str): Ticker
        X_train (numpy array): Training Features
        y_train (numpy array): Training Target
        X_test (numpy array): Testing Features
        y_test (numpy array): Testing Target
        best_config (dictionary): Best Hyperparameters

    Returns:
        All Models (dictionary): Dictionary of Best Model per ticker
        Predicted Return (numpy array): Predicted Return on Testing Set
        MSE (float): Mean Square Error between Target and Prediction
    """
    model = Lstm(input_shape=(X_train.shape[1], X_train.shape[2]),
                 epochs=10,
                 batch_size=best_config['batch_size'],
                 lstm_units=best_config['lstm_units'],
                 dense_units=[best_config['dense_units1'], best_config['dense_units2']],
                 optimizer=best_config['optimizer'])
    model.fit(X_train=X_train, y_train=y_train)
    all_models[ticker] = model # Store the Model for Ticker
    predicted_return = model.model.predict(X_test)
    mse = mean_squared_error(y_test, predicted_return.flatten())
    print(f"For Stock {ticker} the MSE is {mse}")
    return all_models, predicted_return, mse

def run_for_stocks(stock_list, df_per_stock, param_grid, results_folder='../results'):
    """ Function to run the LSTM Model and make predictions for ALL Stocks

    Args:
        stock_list (list): List of Tickers
        df_per_stock (dict): Dictionary of Tickers:DataFrame
        param_grid (dict): Dictionary of Hyperparameters
        results_folder (str, optional): Folder to put CSV Files to. Defaults to '../results'.

    Returns:
        _type_: _description_
    """
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    all_models = {}
    best_configs = []
    for ticker in stock_list:
        print(f"Processing stock: {ticker}")
        df = df_per_stock[ticker].iloc[:, 1:]
        X_train, y_train, X_test, y_test = prepare_data(df)
        #train_size = int(len(X) * 0.8)
        #X_train, y_train = X[:train_size], y[:train_size]
        #X_test, y_test = X[train_size:], y[train_size:]

        exists, csv_path = check_existing_results(ticker, results_folder)
        if exists:
            print(f"Hyperparameter Tuning Results already exist for {ticker}. Loading existing results.")
            try:
                df_to_filter = pd.read_csv("../results/best_configs.csv")
            except FileNotFoundError:
                best_configurations = get_best_configuration(tickers=stock_list)
                best_configurations_df = pd.DataFrame(best_configurations).T.reset_index()
                best_configurations_df.columns = ['ticker',
                                                  'lstm_units',
                                                  'dense_units1',
                                                  'dense_units2',
                                                  'batch_size',
                                                  'optimizer',
                                                  'avg_val_mse']
                best_configurations_df
                best_configurations_df.to_csv("../results/best_configs.csv")
                df_to_filter = pd.read_csv("../results/best_configs.csv")

            try:
                correct_row = df_to_filter[df_to_filter['ticker'] == ticker]
                best_config = correct_row.iloc[0].to_dict()
            except IndexError:
                print(f"Best Configuration for {ticker} not found in best_configuration.csv")
                continue
        else:
            print(f"Running hyperparameter tuning for {ticker}.")
            best_config = run_hyperparameter_tuning(X_train, y_train, param_grid, csv_path)

        best_configs.append({'ticker': ticker, **best_config})

        print(f"Training and evaluating the best model for {ticker}.")
        all_models, predicted_return, mse = train_and_evaluate_best_model(all_models,
                                                                          ticker,
                                                                          X_train,
                                                                          y_train,
                                                                          X_test,
                                                                          y_test,
                                                                          best_config)

        # Save predicted results and MSE
        results = pd.DataFrame({
            'actual': y_test.flatten(),
            'predicted': predicted_return.flatten(),
            'mse': mse
        })

        results.to_csv(os.path.join(results_folder, f"{ticker}_predicted_results.csv"), index=False)
        print(f"Predicted results for {ticker} saved.")

    # Save best configurations for each stock
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv(os.path.join(results_folder, 'best_configs.csv'), index=False)
    print("Best configurations for all stocks saved.")
    return all_models

def check_return_predictions(ticker:str, folder="../results") -> bool:
    return_file = f"{folder}/{ticker}_predicted_results.csv"
    if os.path.exists(return_file):
        return True
    else:
        return False

def get_best_configuration(tickers:list) -> dict:
    """ Go through all hyperparamater tuning files and find best 
        configuration for each Ticker

    Args:
        tickers (list): Tickers

    Returns:
        dict: Dictionary of Best Configuration for each ticker
    """
    best_configurations = {}
    for ticker in tickers:
        try:
            hyperparameter_file = f"../results/{ticker}_hyperparameter_tuning_results.csv"
            df = pd.read_csv(hyperparameter_file)
            min_mse_row = df.loc[df['avg_val_mse'].idxmin()]
            best_config = min_mse_row.to_dict()
            best_configurations[ticker] = best_config
        except:
            continue
    return best_configurations

def final_df_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """ Cleaning the Final Dataframe of the Dataset

    Args:
        df (pd.DataFrame): Dataframe of Features and Target Returns

    Returns:
        pd.DataFrame: Cleaned DataFrame sorted by ticker and datadate (Panel Data)
    """
    df = df.iloc[:, 3:] # Drop the First Three Columns (unnamed, gvkey, iid)
    df = df.sort_values(by=['tic', 'datadate']) # Sort by Ticker and then Date
    df.drop(columns=['conm', 'gsector'], inplace=True) # Drop Company Name and GICS Sector
    df.reset_index(inplace=True) # Reset the Index
    return df

def create_return_arrays(tickers:list, folder='../results') -> np.array:
    """ Function to to take a list of tickers, go to each ticker's predicted returns file


    Args:
        tickers (list): Tickers 
        folder (str, optional): The folder containing the predicted results for each ticker. 
        Defaults to '../results'.

    Returns:
        np.array: _description_
    """
    columns = []
    for ticker in tickers:
        df = pd.read_csv(f"{folder}/{ticker}_predicted_results.csv")
        columns.append(df['predicted'].values)
    return np.array(columns)
