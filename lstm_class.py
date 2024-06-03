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
