from src.GPUusage import configureHardware
from src.DownloadData import downloadStockData
from src.CleanData import cleanData
from src.PrepareData import prepareData
from src.TestDataPrepare import prepareTestData
from src.BuildModel import buildModel
from src.EvaluateModel import evaluateModel
from src.Predict import predict
from src.PlotData import plotPredictions
from src.PredictNextDay import nextDayPrediction




def main(stock):
    configureHardware()
    downloadedData = downloadStockData(stock)
    cleanedData = cleanData(downloadedData)
    xTrain, yTrain, scaler, scaledData, data = prepareData(cleanedData)
    xTest, yTest = prepareTestData(scaledData, data)   
    model = buildModel(xTrain.shape[1:])
    model.fit(xTrain, yTrain, epochs=10, batch_size=32)
    evaluateModel(xTest, yTest, model)
    predictions = predict(xTest, model, scaledData, scaler)
    nextDayPredict = nextDayPrediction(data, model, scaler)
    print(f"Next day predict: {nextDayPredict}")
    plotPredictions(data, predictions)


if __name__ == "__main__":
    stock = "TSLA"
    main(stock)