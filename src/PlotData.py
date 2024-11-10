import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def plotPredictions(data:pd.DataFrame,prediction:np.ndarray):
    trainingDataLen = math.ceil(len(data) * 0.9)
    train = data[:trainingDataLen]
    valid = data[trainingDataLen:].copy()
    valid.loc[:,'Predictions'] = prediction
    plt.figure(figsize=(16, 8))
    plt.title('Model Predictions vs Actual Prices', fontsize=20)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'], label='Training Data', color='blue', linestyle='solid', linewidth=2)
    plt.plot(valid['Close'], label='Actual Prices', color='green', linestyle='solid', linewidth=2)
    plt.plot(valid['Predictions'], label='Predicted Prices', color='red', linestyle='solid', linewidth=2)
    plt.legend(loc='lower right', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
    