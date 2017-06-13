from coinbase.wallet.client import Client
import time, math, datetime, requests
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import json
import pandas as pd

style.use('ggplot')

# now = int(time.time())
now1 = datetime.datetime.now()
then = now1 - datetime.timedelta(days=365)
then = int(time.mktime(then.timetuple()))
now1 = int(time.mktime(now1.timetuple()))
url='https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start='+str(then)+'&end='+str(now1)+'&period=300'
print (url)
r=requests.get(url)
# From Coinbase
api_key = 's1WsCyV7kf8iYcDz'
api_secret = '87YKAW4YDUtUmT4H64CfIs74SGd0dQLa'
# client = Client(api_key,api_secret)
#
# currency_code = 'USD'
#
# price = client.get_spot_price(currency=currency_code)
# # currencies = client.get_currencies()
# # accounts = client.get_accounts()
# # print(accounts)
# print('Current bitcoin price in ' + currency_code + ":" +price.amount)

# print(r.json())
resut = r.json()
df={'high':[],'low':[],'open':[],'close':[],'volume':[],'quoteVolume':[],'weightedAverage':[]}
for i in resut:
    # print(i)
    df['high'].append(i['high'])
    df['low'].append(i['low'])
    df['open'].append(i['open'])
    df['close'].append(i['close'])
    df['volume'].append(i['volume'])
    df['quoteVolume'].append(i['quoteVolume'])
    df['weightedAverage'].append(i['weightedAverage'])
# print(df['High'])
# for data in
df = pd.DataFrame(df)
# print(url)
# print(df.head())
df['HL_PCT'] = (df['high'] - df['low'])/df['low']*100.00000000
df['PCT_Change'] = (df['close'] - df['open'])/df['open']*100.00000000
df = df[['close','HL_PCT','PCT_Change','volume','quoteVolume','weightedAverage']]
# print(df.head())
forecast_col = 'weightedAverage'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
# df.dropna(inplace=True)
print(df.head())

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out:]
df.dropna(inplace=True)

y = np.array(df['label'])
print(len(x), len(y))
#
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size = 0.5)
# # print(x_train)
clf = LinearRegression()
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)
forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy)
