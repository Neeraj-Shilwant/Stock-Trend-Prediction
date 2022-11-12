import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2022-10-27'

st.title("Stock Trend Prediction")
user_input = st.text_input('Enter Stock ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

#Showing data

st.subheader('Data from 2010 - 2022')
st.write(df.describe())

#visualizations
st.subheader('Moving Average 100 Indicator')
fig = plt.figure(figsize=(12,6))
ma100 = df.Close.rolling(100).mean()
plt.plot(df.Close)
plt.plot(ma100,'r')
st.pyplot(fig)

st.subheader('Moving Average 100 vs 200 Indicator')
fig = plt.figure(figsize=(12,6))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig)

#split train test

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1) )
data_train_array = scaler.fit_transform(data_train)

xtrain = []
ytrain = []

for i in range(100,data_train.shape[0]):
    xtrain.append(data_train_array[i-100:i])
    ytrain.append(data_train_array[i,0])
xtrain,ytrain = np.array(xtrain),np.array(ytrain)   

#load model
model = load_model('Keras_LSTM_model.h5')

#testing part
past_100_days = data_test.tail(100)
final_df = past_100_days.append(data_test,ignore_index=True)
input_data = scaler.fit_transform(final_df)

xtest = []
ytest = []
for i in range(100,input_data.shape[0]):
    xtest.append(input_data[i-100:i])
    ytest.append(input_data[i,0])
xtest,ytest = np.array(xtest),np.array(ytest) 

#making predictions
ypred = model.predict(xtest)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
ypred = ypred * scale_factor
ytest = ytest * scale_factor

#final Graph
st.subheader('Prediction Vs Original Trend')
fig2 = plt.figure(figsize=(10,7))
plt.plot(ytest,'b',label = 'Original_price')
plt.plot(ypred,'r',label = 'Predicted_price')
plt.xlabel('Price')
plt.ylabel('Time')
plt.legend()
st.pyplot(fig2)