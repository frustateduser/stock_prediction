def evaluateModel(xTest, yTest, model):
    loss, mse, accuracy, precision = model.evaluate(xTest, yTest)
    print(f"Loss: {loss}, MSE: {mse}, Accuracy: {accuracy}, Precision: {precision}")
