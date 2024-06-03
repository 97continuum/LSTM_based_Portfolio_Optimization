import numpy as np

# Function to predict returns
def predict_returns(models, data, sequence_length=7):
    predicted_returns = []
    for i in range(num_stocks):
        model = models[i]
        stock_data = data[:, i].reshape(-1, 1)
        X_test, _ = create_sequences(stock_data, sequence_length)
        predicted_return = model.model.predict(X_test)
        predicted_returns.append(predicted_return.flatten())
    return np.array(predicted_returns).T