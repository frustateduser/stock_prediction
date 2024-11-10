import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

def prepareData(data:pd.DataFrame):
    trainingDataLen = math.ceil(len(data) * 0.9)
    dataset = data.values
    scaler = MinMaxScaler() # or can use StandardScaler()
    scaledData = scaler.fit_transform(dataset)
    trainData  = scaledData[:trainingDataLen]
    xTrain, yTrain = [], []
    for i in range(100, len(trainData)):
        xTrain.append(trainData[i-100:i, 0])
        yTrain.append(trainData[i, 0])
    
    xTrain = np.array(xTrain).reshape(-1, 100, 1)
    yTrain = np.array(yTrain)
    return xTrain, yTrain, scaler, scaledData, data