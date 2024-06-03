"""
    Create Class for LSTM    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # type: ignore
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

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
                 batch_size=32) -> None:
 
        # Do this because linter was telling me [] cannot be a default value
        if dense_units is None:
            dense_units = [100,50,1]

        self.input_shape=input_shape
        self.lstm_units=lstm_units
        self.dense_units=dense_units
        self.optimizer=optimizer
        self.loss=loss
        self.epochs=epochs
        self.batch_size=batch_size
        self.history=None
        self.model=None
        return None

    def build_model(self):
        """ Build the LSTM Model

        Returns:
            None: None
        """
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(LSTM(self.lstm_units))
        for units in self.dense_units[:-1]:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(self.dense_units[-1]))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return None
    
    def summary(self):
        """ Return LSTM RNN Model Summary

        Returns:
            Summary: Model Summary
        """
        return self.model.summary()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """ Fit the Model

        Args:
            X_train (Numpy Array): Training Features
            y_train (Numpy Array): Training Target
            X_val (Numpy Array, optional): Validation Features. Defaults to None.
            y_val (Numpy Array, optional): Validation Target. Defaults to None.
        """
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

    def plot_performance(self):
        """ Plot the Training vs Validation Performance

        Returns:
            Figure: Loss v/s Epochs
        """
        if self.history is None:
            print("No Training History found, please train the Model first.")
            return None
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'], label="Training Loss")
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("model_performance.png", dpi=300)
        plt.show()
