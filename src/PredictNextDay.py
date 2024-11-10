import numpy as np
import pandas as pd


def nextDayPrediction(data:pd.DataFrame, model, scaler):
    last100Days = data[-100:].values
    last100Days = scaler.transform(last100Days)
    x = np.array(last100Days).reshape(-1, 100, 1)
    prediction = model.predict(x)
    prediction = prediction.reshape(-1, 1)
    dummy_data = np.zeros((prediction.shape[0], data.shape[1]))  
    dummy_data[:, -1] = prediction[:, 0]
    prediction = scaler.inverse_transform(dummy_data)[:,-1]
    return prediction