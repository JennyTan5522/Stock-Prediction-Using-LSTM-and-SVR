Machine Learning Stock Prediction/Foreasting in Python Using LSTM and SVR

About:
- The goal of this project aims to develop a model with high accuracy to predict the stock market and make statistically informed stock trading decisions.
- Model prediction provides us insights into which companies' stocks are worth investing in so that risk of losing money can be minimized, as well as help an individual or company to make more informed decisions on stock investing.

Dataset:
- Two datasets are selected from Kaggle, which are Johnson & Johnson (JNJ) and Exxon Mobil Corporation (XOM) [https://www.kaggle.com/datasets/rprkh15/sp500-stock-prices?select=MSI.csv]
- Each of the datasets consists of 8 columns which are the features, date, open, high, low, close, volume, dividends and stock splits and **15236** rows (Date range from 1962-01-02 to 2022-07-12)
- Data Features:
  - Date: The date is in the format yy-mm-dd
  - Open: Price of the stock when the market opens
  - High: Highest price reached in the day
  - Low: Lowest price reached in the day
  - Close: Price of the stock when the market closes
  - Volume: Number of shares traded in a day
  - Dividends: The dividends of the stock
  - Stock Splits: The stock splits of the company. In a stock split, a company divides its existing stock into multiple shares to boost liquidity.

Data Preparation:
1. EDA
   - Average stock price movement for JNJ (Johnson & Johnson) and XOM (Exxon Mobil Corp) over a roughly 60-year period from 1962 to 2022. To observe the overall trend of each stock, we first plotted line charts for each based on the average column. Since 1990, the XOM and JNJ stock prices have been rising. JNJ's average stock value began to be higher than XOM's, but this changed after XOM's abrupt, dramatic growth between 2000 and 2010. However, the volume parameter of the stocks is extremely erratic and lacks any sort of discernible pattern. Inclination of the stock JNJ began to rise sharply after hitting its low point in the late 1990s. Not able to obtain any useful insights from the volume chart.

     ![image](https://github.com/user-attachments/assets/b20b543b-30d1-4132-a4b2-35ff7ff8644a)

     ![image](https://github.com/user-attachments/assets/437a2630-229a-40e8-a46e-7f629785458a)

- 10-day, 100-day, and 365-day moving averages of each stock has been determined to identify the trends.

  ![image](https://github.com/user-attachments/assets/2b531e7a-70b4-45b9-9f0e-557a0e7585a4)

  ![image](https://github.com/user-attachments/assets/044c2256-9db8-43f9-9688-2842b505cc14)

- Based on the last four historical data points, JNJ's risk and return performance shows a minimum risk of 1.2% but surpasses it with a higher return (about 0.048% return). In contrast, the XOM exhibits a high risk but low expected return (about 0.041% return and 1.8% risk). Lower risk and greater return lead to the conclusion that JNJ is the more effective of the two (i.e., worth investing).

  ![image](https://github.com/user-attachments/assets/e7a3a791-3313-43d4-a4cf-ef4329f4f09c)

- Association between the XOM and JNJ has been evaluated on a heatmap. It displays an extremely high correlation of 0.88. In other words, it indicates that XOM and JNJ are likely to have a similar business nature.
  ![image](https://github.com/user-attachments/assets/5401b9f2-8d64-469c-9aa8-6cea1545cc71)


3. Check Missing Values to ensure data quality and data accuracy
   ![image](https://github.com/user-attachments/assets/2b2af2af-a2d2-49b0-81f7-499439e58b30)

   
5. Feature Selection (Using Pearson Correlation)
   - Since only Open, High , Low , Close (OHLC) have a high correlation , we will be transform OHLC into new variable 'Average'

Model using predictive model:
1. SVR
2. LSTM

Model Evaluation:
- Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).

Advanced Analysis
- Prediction for next 30 days using LSTM.

Results:
- LSTM obtained highest accuracy as compared to SVR.
- LSTM model is used to perform forecasting on the full JNJ dataset and the RMSE values obtained for the training set and testing set are 0.2167 and 2.3747 whereas 
the MAPE values obtained are 1.7414% and 1.2100%. LSTM is also used for forecasting the average stock prices of the full XOM dataset and the training set obtained an RMSE value of 0.2550 and 0.9429 for the test set. 
- The RMSE of the SVR model of the training set and testing set are 14.3198 and 19.8299 respectively whereas the MAPE values are 66.8251% and 27.4985%. 
