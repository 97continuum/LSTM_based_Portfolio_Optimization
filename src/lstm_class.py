"""
    Create Class for LSTM    
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from scikeras.wrappers import KerasRegressor # type: ignore
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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

    def build_model(self):
        """ Build the LSTM Model

        Returns:
            None: None
        """
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(LSTM(self.lstm_units, return_sequences=True))
        model.add(LSTM(self.lstm_units))
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(1)) # One output unit
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

    def summary(self):
        """ Return LSTM RNN Model Summary

        Returns:
            Summary: Model Summary
        """
        model = self.build_model()
        return model.summary()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """ Fit the Model

        Args:
            X_train (Numpy Array): Training Features
            y_train (Numpy Array): Training Target
            X_val (Numpy Array, optional): Validation Features. Defaults to None.
            y_val (Numpy Array, optional): Validation Target. Defaults to None.
        """
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

    def plot_performance(self):
        """ Plot the Training vs Validation Performance

        Returns:
            Figure: Loss v/s Epochs
        """
        if self.history is None:
            print("No Training History found, please train the Model first.")
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label="Training Loss")
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("model_performance.png", dpi=300)
        plt.show()

    def hyperparameter_tuning(self, X, y, param_grid:dict, cv_splits=5):
        """ Use Grid Search and Time Series Cross Validation for Hyperparameter tuning. 

        Args:
            X (Numpy 2D Array): _description_
            y (Numpy Array): _description_
            param_grid (Dictionary): _description_
            cv_splits (int, optional): _description_. Defaults to 5.
        """
        def build_keras_model(lstm_units=100,
                              dense_units1=100,
                              dense_units2=50,
                              optimizer='adam'):
            """ Build Keras Model for Time Series Cross Validation

            Args:
                lstm_units (int, optional): Units for LSTM Layer. Defaults to 100.
                dense_units1 (int, optional): Units for First Dense Layer. Defaults to 100.
                dense_units2 (int, optional): Units for Second Dense Layer. Defaults to 50.
                optimizer (str, optional): Adam Optimizer. Defaults to 'adam'.

            Returns:
                Model: LSTM Model
            """
            model = Sequential()
            model.add(Input(shape=self.input_shape)) # Input Layer
            model.add(LSTM(lstm_units)) # LSTM Cell
            model.add(Dense(dense_units1, activation='relu')) # Hidden Layer
            model.add(Dense(dense_units2, activation='relu')) # Hidden Layer
            model.add(Dense(1)) # Output Layer
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model

        model = KerasRegressor(build_fn=build_keras_model,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               verbose=False)
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=tscv,
                            scoring='neg_mean_squared_error')
        grid_result = grid.fit(X, y)

        best_config = grid_result.best_params_
        print(f"Best Performance got a score of {grid_result.best_score_}")
        print(f"Best Parameters were: {best_config}")
        return best_config
