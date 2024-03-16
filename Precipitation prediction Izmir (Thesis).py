#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np 


# In[30]:


import pandas as pd 


# In[31]:


data = pd.read_csv('Urfa.csv', index_col="DATE")
data


# In[32]:


data.index = pd.to_datetime(data.index)


# In[33]:


data["percip"].plot()


# In[34]:


data.index.year.value_counts().sort_index()


# In[35]:


import matplotlib.pyplot as plt 
from datetime import datetime


# In[36]:


data.index = pd.to_datetime(data.index)
data 


# In[37]:


plt.figure(figsize=(10,4))
plt.plot(data)


# In[10]:


pip install pmdarima 


# In[11]:


data.head()


# In[12]:


data.tail()


# In[13]:


data.shape


# In[14]:


data['percip'].plot(figsize=(12,5))


# In[15]:


from statsmodels.tsa.stattools import adfuller 


# In[16]:


def ad_test (data):
    dftest = adfuller(data, autolag='AIC')
    print("1. ADF : ",dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags :", dftest[2])
    print("4. Num Of Observation Used for ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)


# In[17]:


ad_test(data['percip'])


# #when the p-value is small it means that our data is stasionary 

# # Figure Out for ARIMA Model 

# In[18]:


from pmdarima import auto_arima 


# In[20]:


#Ignore harmless warnings
import warnings 
warnings.filterwarnings("ignore")


# In[21]:


stepwise_fit = auto_arima(data['percip'], trace=True, suppress_warnings=True)
stepwise_fit.summary()


# In[22]:


from statsmodels.tsa.arima_model import ARIMA


# In[10]:


print(data.shape)


# In[11]:


train = data.iloc[:2525]
train 


# In[12]:


test = data.iloc[2525:]
test


# # Train the model 

# In[31]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train['percip'],order=(2,0,1))
model = model.fit()
model.summary()


# In[33]:


start = len(train)
end = len(train)+len(test)-1
pred=model.predict(start=start,end=end,typ='levels')
pred.index = data.index[start:end+1]
print(pred)


# In[34]:


pred.plot(legend=True)
test['percip'].plot(legend=True)


# In[35]:


test['percip'].mean()


# In[36]:


from sklearn.metrics import mean_squared_error
from math import sqrt 
rmse = sqrt(mean_squared_error(pred,test['percip']))
print(rmse)


# In[37]:


model2 = ARIMA(data['percip'],order=(2,0,1))
model2=model2.fit()
data.tail()


# In[43]:


index_future_dates = pd.date_range(start='2022-01-01',end='2023-01-01')
pred = model2.predict(start=len(data),end=len(data)+365,typ='levels').rename('ARIMA Predictions')
pred.index=index_future_dates
print(pred)


# In[44]:


pred.plot(figsize=(12,5),legend=True)


# In[46]:


import tensorflow as tf 
import os 


# In[76]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[78]:


data['percip']=scaler.fit_transform(data[['percip']])


# In[81]:


data


# In[82]:


train = data.iloc[:2525]
train 


# In[83]:


test = data.iloc[2525:]
test


# In[84]:


#Create sequence for LSTM 
seq_length = 10


# In[85]:


def create_sequences(data, seq_length):
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length]
        target = data.iloc[i + seq_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


# In[86]:


X_train, y_train = create_sequences(train, seq_length)
X_test, y_test = create_sequences(test, seq_length)


# In[87]:


y_train


# In[88]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[89]:


# Build and train LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=15, batch_size=32)


# In[90]:


# Prepare input for prediction
last_sequence = data.iloc[-seq_length:].values.reshape(1, seq_length, 1)


# In[91]:


# Generate predictions for the next 365 days
predictions = []

for _ in range(365):
    next_pred = model.predict(last_sequence)
    predictions.append(next_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0, -1, 0] = next_pred


# In[92]:


predictions


# In[93]:


# Inverse scaling
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


# In[95]:


# Create index for predictions
next_days = pd.date_range(start=data.index[-1] + pd.DateOffset(days=1), periods=365)


# In[96]:


# Create DataFrame of predictions
predictions_df = pd.DataFrame({'Date': next_days, 'Predicted_precip': predictions})
predictions_df.set_index('Date', inplace=True)


# In[114]:


predictions_df['']


# In[98]:


# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data.index, scaler.inverse_transform(data[['percip']]), label='Actual Data')
plt.plot(predictions_df.index, predictions_df['Predicted_precip'], label='Predicted Data', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.legend()
plt.show()


# In[99]:


# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Plot the actual data with a solid line
plt.plot(data.index, scaler.inverse_transform(data[['percip']]), label='Actual Data', linestyle='-', marker='o')

# Plot the predicted data with a dashed line
plt.plot(predictions_df.index, predictions_df['Predicted_precip'], label='Predicted Data', linestyle='--', marker='x')

plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.legend()
plt.show()


# In[38]:


data


# In[39]:


train 


# In[15]:


pip install prophet 


# In[40]:


data


# In[41]:


new_data = data.reset_index()[["DATE", "percip"]].rename(
    columns={"DATE": "ds", "percip": "y"}
)
new_data


# In[42]:


from prophet import Prophet 

model = Prophet()
#Fit the model 
model.fit(new_data)

#create date to predict 
future_dates = model.make_future_dataframe(periods=365)

#make predictions
predictions = model.predict(future_dates)

predictions.head()


# In[43]:


predictions.tail()


# In[44]:


model.plot(predictions)


# In[45]:


model.plot_components(predictions)


# In[46]:


from prophet.diagnostics import cross_validation, performance_metrics

# Perform cross-validation with initial 365 days for the first training data and the cut-off for every 180 days.

df_cv = cross_validation(model, initial='365 days', period='180 days', horizon = '365 days')

# Calculate evaluation metrics
res = performance_metrics(df_cv)

res


# In[47]:


from prophet.plot import plot_cross_validation_metric
#choose between 'mse', 'rmse', 'mae', 'mape', 'coverage'

plot_cross_validation_metric(df_cv, metric= 'rmse')


# In[49]:


from prophet.plot import plot_cross_validation_metric
#choose between 'mse', 'rmse', 'mae', 'mape', 'coverage'

plot_cross_validation_metric(df_cv, metric= 'mse')


# In[50]:


data


# In[ ]:




