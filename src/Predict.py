import numpy as np

def predict(xTest, model,scaledData,scaler):
    predictions = model.predict(xTest)
    predictions = predictions.reshape(-1, 1)
    dummy_data = np.zeros((predictions.shape[0], scaledData.shape[1]))  
    dummy_data[:, -1] = predictions[:, 0]
    predictions = scaler.inverse_transform(dummy_data)[:,-1]
    return predictions