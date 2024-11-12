# Stock Prediction

This project implements a stock prediction model using Long Short-Term Memory (LSTM) neural networks. The model is designed to predict future stock prices based on historical data and various technical indicators.

---

### Project Description

The goal of this project is to build a predictive model that can forecast stock prices. The model uses LSTM, a type of recurrent neural network (RNN) that is well-suited for time series forecasting tasks. The project includes data preprocessing steps, feature engineering, model training, and evaluation.

### Features

- **Data Collection**: Fetches historical stock data using the `yfinance` library.
- **Data Preprocessing**: Cleans and prepares the data, including calculating technical indicators such as SMA, RSI, Bollinger Bands, MACD, and Williams %R.
- **Model Training**: Trains an LSTM model to predict future stock prices.
- **Evaluation**: Evaluates the model's performance using metrics such as loss, mean squared error (MSE), accuracy, and precision.
- **Visualization**: Plots the actual vs predicted stock prices for visual comparison.

---

### Setting up the Project

>Follow the steps below to set up the project:
>
>1. ***Clone the Repository***:
>
>```sh
>git clone https://github.com/frustateduser/stock_prediction.git
>cd stock_prediction
>```
>
>2. ***Set Up the Virtual Environment***: Create a virtual environment to >manage dependencies. [Refer to the official documentaion for more details.]>(https://docs.python.org/3/library/venv.html)
>
>```sh
> python -m venv venv
>```
>
>3.***Activate the virtual Environment***:
>
>-**Windows**:
>```sh
>venv\Scripts\activate
>```
>
>-**Unix or MacOS**
>```sh
>source venv/bin/activate
>```
>4.***Install Dependencies***: Install the required package using `pip`:
>```sh
>pip install -r requirments.txt
>```
>
>
---

### Usage

1.Data Preparation:
- Fetch historical stock data using the yfinance library.
- Preprocess the data and calculate technical indicators.

2.Model Training: 

- Train the LSTM model using the preprocessed data.

3.Model Evaluation:

- Evaluate the model's performance using various metrics.

4.Prediction and Visualization:

- Use the trained model to make predictions.
- Plot the actual vs predicted stock prices.

---

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgements
- [yfinance](https://github.com/ranaroussi/yfinance) for fetching historical stock data.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for building and training the LSTM model.
- [scikit-learn](https://scikit-learn.org/) for data preprocessing utilities.
- [ta](https://github.com/bukosabino/ta) for technical analysis indicators.



![output graph for close price of TSLA(tesla) stocks](output.png "output graph for close price of TSLA(tesla) stocks")