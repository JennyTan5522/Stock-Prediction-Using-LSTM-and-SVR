Machine Learning Stock Prediction/Foreasting in Python Using LSTM and SVR

About:
- The goal of this project aims to develop a model with high accuracy to predict the stock market and make statistically informed stock trading decisions.
- Model prediction provides us insights into which companies' stocks are worth investing in so that risk of losing money can be minimized, as well as help an individual or company to make more informed decisions on stock investing.

Dataset:
- Two datasets are selected from Kaggle, which are Johnson & Johnson (JNJ) and Exxon Mobil Corporation (XOM) [https://www.kaggle.com/datasets/rprkh15/sp500-stock-prices?select=MSI.csv]

Data Preparation:
1. Data Features
  - Date: The date is in the format yy-mm-dd
  - Open: Price of the stock when the market opens
  - High: Highest price reached in the day
  - Low: Lowest price reached in the day
  - Close: Price of the stock when the market closes
  - Volume: Number of shares traded in a day
  - Dividends: The dividends of the stock
  - Stock Splits: The stock splits of the company. In a stock split, a company divides its existing stock into multiple shares to boost liquidity.

2. Check Missing Values to ensure data quality and data accuracy
3. Feature Selection (Using Pearson Correlation)
   - Since only Open, High , Low , Close (OHLC) have a high correlation , we will be transform OHLC into new variable 'Average'
4. EDA

Model using predictive model:
1. SVR
2. LSTM

Model Evaluation:
- Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).

Advanced Analysis
- Prediction for next 30 days.

The Stacked LSTM model is used to perform forecasting on the full JNJ dataset and the RMSE values obtained for the training set and testing set are 0.2167 and 2.3747 whereas 
the MAPE values obtained are 1.7414% and 1.2100%. Stacked LSTM is also used for forecasting the average stock prices of the full XOM dataset and the training set obtained an RMSE value of 0.2550 and 0.9429 for the test set. The RMSE of the SVR model of the training set and testing set are 14.3198 and 19.8299 respectively whereas the MAPE values are 
66.8251% and 27.4985%. 
