"""
    Create Class for LSTM    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import TimeSeriesSplit

class Lstm():
    """ LSTM Class for Return Prediction
    """
    def __init__(self,
                 input_shape,
                 lstm_units=100,
                 dense_units=None,
                 optimizer='adam',
                 loss='mean_squared_error',
                 epochs=10,
                 batch_size=32,
                 dropout_rate=0.2,
                 regularization_value=0.01) -> None:

        # Do this because linter was telling me [] cannot be a default value
        if dense_units is None:
            dense_units = [100,50]

        self.input_shape=input_shape
        self.lstm_units=lstm_units
        self.dense_units=dense_units
        self.optimizer=optimizer
        self.loss=loss
        self.epochs=epochs
        self.batch_size=batch_size
        self.history=None
        #self.model=None
        self.dropout_rate = dropout_rate
        self.regularization_value = regularization_value

    def build_model(self, lstm_units=100, dense_units1=100, dense_units2=50, optimizer='adam'):
        """ Build the LSTM Model """
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(LSTM(lstm_units, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        model.add(LSTM(lstm_units))

        regularizer = l2(self.regularization_value)

        model.add(Dense(self.dense_units[0], activation='relu', kernel_regularizer=regularizer))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())

        model.add(Dense(self.dense_units[1], activation='relu', kernel_regularizer=regularizer))

        model.add(Dense(1))  # One output unit

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mse'])
        return model

    def summary(self):
        """ Return LSTM RNN Model Summary

        Returns:
            Summary: Model Summary
        """
        model = self.build_model()
        return model.summary()

def fit(self, X_train, y_train, X_val=None, y_val=None, dropout_rate=0.2, regularization_value=0.01):
    """ Fit the Model

    Args:
        X_train (Numpy Array): Training Features
        y_train (Numpy Array): Training Target
        X_val (Numpy Array, optional): Validation Features. Defaults to None.
        y_val (Numpy Array, optional): Validation Target. Defaults to None.
        dropout_rate (float, optional): Dropout rate for dropout layers. Defaults to 0.2. 
        regularization_value (float, optional): Regularization value for L2 regularization. Defaults to 0.01.
    """
    self.dropout_rate = dropout_rate  
    self.regularization_value = regularization_value 
    self.model = self.build_model()
    if X_val is not None and y_val is not None:
        self.history = self.model.fit(X_train,
                                      y_train,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      validation_data=(X_val, y_val)
                                      )
    else:
        self.history = self.model.fit(X_train,
                                      y_train,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size
                                      )

    def hyperparameter_tuning(self, X, y, param_grid, epochs=10, n_splits=5, csv_path='grid_search_results.csv'):
        """ Grid Search to find best Hyperparameters with Time Series Cross-Validation

        Args:
            X (Array-like): Features
            y (Array-like): Target Variable
            param_grid (Dictionary): Grid of Hyperparameters
            epochs (int, optional): Number of Epochs. Defaults to 10.
            n_splits (int, optional): Number of splits for time series cross-validation. Defaults to 5.
            csv_path (str, optional): Path to save the CSV file. Defaults to 'grid_search_results.csv'.

        Returns:
            best_params (dict): Best Parameters for the Neural Network Model
            best_history (History): Training history of the best model
        """
        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # DataFrame to store the parameters and their average validation MSE
        results = []

        best_score = float('inf')
        best_params = None

        for batch_size in param_grid['batch_size']:
            for lstm_units in param_grid['lstm_units']:
                for dense_units1 in param_grid['dense_units1']:
                    for dense_units2 in param_grid['dense_units2']:
                        for optimizer_name in param_grid['optimizer']:
                            for dropout_rate in param_grid['dropout']:
                                for regularization_value in param_grid['regularization_value']:

                                    current_params = {
                                        'lstm_units': lstm_units,
                                        'dense_units1': dense_units1,
                                        'dense_units2': dense_units2,
                                        'batch_size': batch_size,
                                        'optimizer': optimizer_name,
                                        'dropout_rate': dropout_rate,
                                        'regularization_value': regularization_value
                                    }

                                    print(f"Current Parameters: {current_params}")

                                    val_scores = []

                                    for train_index, val_index in tscv.split(X):
                                        X_train, X_val = X[train_index], X[val_index]
                                        y_train, y_val = y[train_index], y[val_index]

                                        # Debug: Print shapes of the splits
                                        #print(f"Train shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
                                        #print(f"Val shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")

                                        # Check if training or validation sets are empty
                                        if X_train.size == 0 or X_val.size == 0 or y_train.size == 0 or y_val.size == 0:
                                            print("Skipping due to empty training or validation set")
                                            continue

                                        # Build and compile model with current parameters
                                        self.dropout_rate = dropout_rate
                                        self.regularization_value = regularization_value
                                        self.model = self.build_model(lstm_units, dense_units1, dense_units2, optimizer_name)
                                        history = self.model.fit(X_train,
                                                                y_train,
                                                                epochs=epochs,
                                                                batch_size=batch_size,
                                                                validation_data=(X_val, y_val),
                                                                verbose=0)
                                        val_mse = history.history['val_mse'][-1]
                                        val_scores.append(val_mse)

                                    avg_val_mse = np.mean(val_scores)

                                    # Add the parameters and their average validation MSE to the results list
                                    results.append({
                                        'lstm_units': lstm_units,
                                        'dense_units1': dense_units1,
                                        'dense_units2': dense_units2,
                                        'batch_size': batch_size,
                                        'optimizer': optimizer_name,
                                        'dropout_rate': dropout_rate,
                                        'regularization_value': regularization_value,
                                        'avg_val_mse': avg_val_mse
                                    })

                                    if avg_val_mse < best_score:
                                        best_score = avg_val_mse
                                        best_params = current_params

                                    print(f"Params: {current_params}, Average Validation MSE: {avg_val_mse:.4f}")

                                    # Export results to a CSV file
                                    results_df = pd.DataFrame(results)
                                    results_df.to_csv(f"../results/{csv_path}", index=False)
                                    print(f"Results saved to {csv_path}")

        print(f"Best Params: {best_params}, Best Validation MSE: {best_score:.4f}")
        return best_params
    