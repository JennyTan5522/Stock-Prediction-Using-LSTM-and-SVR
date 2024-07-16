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

  - Based on the last four historical data points, JNJ's risk and return performance shows a minimum risk of 1.2% but surpasses it with a higher return (about 0.048% return). In contrast, the XOM exhibits a high risk but low expected return   (about 0.041% return and 1.8% risk). Lower risk and greater return lead to the conclusion that JNJ is the more effective of the two (i.e., worth investing).
  
    ![image](https://github.com/user-attachments/assets/e7a3a791-3313-43d4-a4cf-ef4329f4f09c)

    ![image](https://github.com/user-attachments/assets/c24daeab-60cc-42c9-8d0f-50ebddcf8c24)
  
  - Association between the XOM and JNJ has been evaluated on a heatmap. It displays an extremely high correlation of 0.88. In other words, it indicates that XOM and JNJ are likely to have a similar business nature.
    ![image](https://github.com/user-attachments/assets/5401b9f2-8d64-469c-9aa8-6cea1545cc71)


3. Check Missing Values to ensure data quality and data accuracy
   
      ![image](https://github.com/user-attachments/assets/2b2af2af-a2d2-49b0-81f7-499439e58b30)

   
5. Feature Selection (Using Pearson Correlation)
     - From previous research, it is shown that for trend detection of stock prices, the Open High Low Close (OHLC) levels have high predictive potentials and are easier to predict compared to the traditional Close price.
     - From the Pearson’s correlation coefficient performed for the JNJ and XOM datasets, at the threshold value of 0.9, the open, high, low and close attributes are highly correlated to each other, thus, the four columns will be selected as the target by transforming into an average which will be done in the data transformation step. The remaining variables, volume, stock splits and dividend are dropped from the dataset as they do not have high correlations to the target. 

     ![image](https://github.com/user-attachments/assets/0bf80675-3a8a-4b73-94c8-318588a11a76)

6. Feature Scaling
   - Scales the value of a variable to a value between 0 and 1 to achieve a higher precision. Higher precision is achieved by feature scaling because the values of the data are not spread out in a wide range.
   - The computational cost and memory consumption of the data in the dataset will also decrease when data with large value is reduced.
  
   - Before Scaling:
     
     ![image](https://github.com/user-attachments/assets/611bf9ae-e992-47bb-825f-51ee4f3ea8d1)

   - After scaling:

     ![image](https://github.com/user-attachments/assets/90d1eabf-0dae-41af-88d0-539d2caeb611)

7. Model Splitting
   - Before performing the train test split, copies of the JNJ and XOM datasets are made so that the full dataset can be retained while a portion of the dataset that have only stock prices of the latest 4 years can be obtained, resulting in 2 datasets, 1 full and 1 partial for the dataset of each company.
   - The last 30 days of data are excluded from the dataset so that the data can be treated as extrapolation and validation can be performed for the extrapolated results.
   - The 4 datasets are then split into train and test datasets and due to the data being time series data, sequence is important because historical data will be used to forecast the future data. Therefore, the train and test sets are not split randomly, instead, they are split according to the sequence, the first 80% of the data will form the train set and the last 20% will form the test set. 


Model using predictive model:
1. Stacked LSTM:
   - 100 hidden neurons and use Rectified Linear Unit activation (ReLu) for the activation function.
   - Number of epochs used to train the model is set to 100 and a batch size of 64.
   - This model can handle sequential data, and it is able to memorize and consider previous inputs and outputs when performing prediction.
  
2. SVR:
   - Find best parameter values using GridSearchCV
   -  XOM Full C=5, XOM Partial C=1000; JNJ C=20
   -  The Gamma value of the SVR model used for all the datasets is set to 0.001 except for the JNJ partial dataset that is set to 0.01.
   -  Kernel for XOM=Linear, JNJ=RBF
   -  Degree=2

Model Evaluation:
- Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).

Advanced Analysis
- Prediction for next 30 days using LSTM.

Results:
JNJ Dataset
1) Stacked LSTM: Full JNJ dataset and the RMSE values obtained for the training set and testing set are 0.2167 and 2.3747 whereas the MAPE values obtained are 1.7414% and 1.2100%.

  ![image](https://github.com/user-attachments/assets/a33711e1-52bc-4cd8-b9b7-4b41d448ea76)

  SVR: Full JNJ dataset and the RMSE values obtained for the training set and testing set are 14.3198 and 19.8299 whereas the MAPE values obtained are 66.8251% and 27.4985%.
  
  ![image](https://github.com/user-attachments/assets/a0f32c03-0b95-4e04-bd6e-3a56b967241e)

2) Stacked LSTM: Partial JNJ dataset and the RMSE values obtained for the training set and testing set are 1.6375 and 1.8993 whereas the MAPE values obtained are 0.8517% and 0.8849%.

   ![image](https://github.com/user-attachments/assets/434a6456-c808-40bf-9004-6b533926e30d)

  SVR: Partial JNJ dataset and the RMSE values obtained for the training set and testing set are 3.4937 and 3.7485 whereas the MAPE values obtained are 2.0837% and 1.7861%.

  ![image](https://github.com/user-attachments/assets/11697ab5-6315-45ef-9630-726c8e075a1f)

3) Stacked LSTM: Forecasting for the Full JNJ dataset
   
   ![image](https://github.com/user-attachments/assets/51b8e679-de5e-48ee-bf4a-9ac8b11b70b1)

   SVR: Forecasting for the Full JNJ dataset

   ![image](https://github.com/user-attachments/assets/0db77e88-0621-45f2-b32d-84806447857f)

4) SVR: Forecasting for the Partial JNJ dataset

   ![image](https://github.com/user-attachments/assets/d9062f75-6b9d-48c3-bd28-50264766b9f8)

   SVR: Forecasting for the Partial JNJ dataset
   ![image](https://github.com/user-attachments/assets/10408a3a-63fe-4d68-998d-7875b92997c4)

XOM Dataset
1) Stacked LSTM: Full XOM

  ![image](https://github.com/user-attachments/assets/40e62ca0-a1aa-49a7-b0dd-b87ff232c465)

   SVR: Full XOM

   ![image](https://github.com/user-attachments/assets/2c26c44e-f403-4352-8080-9b0374bf5a6a)

2)  SVR: Forecasting Full XOM

   ![image](https://github.com/user-attachments/assets/7a1dab5a-98ba-4ce8-a722-39dab7ef2029)


3) SVR: Forecasting Partial XOM

   ![image](https://github.com/user-attachments/assets/b3da2465-7ab8-4e24-b9b6-e2e25c35075b)

Plots:
Actual and forecasted values by Stacked LSTM model for the partial JNJ dataset

![image](https://github.com/user-attachments/assets/cdf1cf85-ed05-4b93-855b-cab7a006e04f)

Actual and forecasted values by SVR model for the partial JNJ dataset

![image](https://github.com/user-attachments/assets/869943c9-08b1-42e9-b0e1-3d580ac31e7d)

Actual and forecasted values by Stacked LSTM model for the partial XOM dataset

![image](https://github.com/user-attachments/assets/7ae9680d-477f-40e9-a218-e6a25ba376b0)

Actual and forecasted values by SVR model for the partial XOM dataset

![image](https://github.com/user-attachments/assets/0ba74206-e1f3-4a61-ad37-fbb4b901faad)

Actual and forecasted 30 days of extrapolated data by Stacked LSTM model for the partial JNJ dataset

![image](https://github.com/user-attachments/assets/a023b386-8d82-4e66-a184-8dc0e54912f7)
   
 Actual and forecasted 30 days of extrapolated data by Stacked LSTM model for the partial XOM dataset
 
![image](https://github.com/user-attachments/assets/720f1f9a-cec4-477f-8eda-a2e2e3bcb1d8)

Conclusion:
- LSTM obtained highest accuracy as compared to SVR.
