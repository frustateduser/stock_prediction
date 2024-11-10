import math
import pandas as pd
import numpy as np


def prepareTestData(scaledData: np.ndarray, data:pd.DataFrame):
    trainingDataLen = math.ceil(len(data) * 0.9)
    testData  = scaledData[trainingDataLen -100:]
    xTest, yTest = [], []
    for i in range(100, len(testData)):
        xTest.append(testData[i-100:i, 0])
    
    xTest = np.array(xTest).reshape(-1, 100, 1)
    yTest = testData[100:, 0]  
    
    return xTest, yTest
