from keras import Sequential,Input
from keras.layers import LSTM,Dense,Bidirectional,Conv1D
 
def buildModel(inputShape:tuple):
    model = Sequential()
    model.add(Input(shape=inputShape))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(LSTM(100, return_sequences=False, dropout=0.2))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'accuracy', 'precision']) 
    return model