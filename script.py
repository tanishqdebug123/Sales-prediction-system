import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


store_sales = pd.read_csv('train.csv')
# print(store_sales.head(10))
# print(store_sales.info())

store_sales = store_sales.drop(['store','item'], axis=1)


store_sales['date'] = pd.to_datetime(store_sales['date'])
store_sales['date'] = store_sales['date'].dt.to_period('M')
monthly_sales = store_sales.groupby('date').sum().reset_index()

monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()

#print(monthly_sales.head(10))

# Data plotting 


# plt.figure(figsize=(15,5))
# plt.plot(monthly_sales['date'], monthly_sales['sales'])
# plt.xlabel('Date')
# plt.xlabel('Sales')
# plt.title("Monthly Customer Sales")
# plt.show()


monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)

#making the data stationary
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)


#Stationary data plotting


# plt.figure(figsize=(15,5))
# plt.plot(monthly_sales['date'], monthly_sales['sales_diff'])
# plt.xlabel('Date')
# plt.xlabel('Sales')
# plt.title("Monthly Customer Sales Diff")
# plt.show()


supverised_data = monthly_sales.drop(['date','sales'], axis=1)

for i in range(1,13):
    col_name = 'month_' + str(i)
    supverised_data[col_name] = supverised_data['sales_diff'].shift(i)
supverised_data = supverised_data.dropna().reset_index(drop=True)
supverised_data.head(10)

train_data = supverised_data[:-12]
test_data = supverised_data[-12:]

# print('Train Data Shape:', train_data.shape)
# print('Test Data Shape:', test_data.shape)


scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

X_train, y_train = train_data[:,1:], train_data[:,0:1]
X_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()


print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_test Shape:', X_test.shape)
print('y_test Shape:', y_test.shape)



sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

act_sales = monthly_sales['sales'][-13:].to_list()

print(predict_df)

