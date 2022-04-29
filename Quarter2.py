
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime
from pandas import to_datetime
from statsmodels.tsa.holtwinters  import ExponentialSmoothing 
# version for statsmodel '0.13.2'

url = 'ForecastApril.csv'
df = pd.read_csv(url, index_col = 'Date', parse_dates=True)
df.info()


df.head()
df.index.freq = 'D'

df.plot(figsize=(12,8))

train=df.iloc[:1155]
test= df.iloc[1155:]


#1.023 is the weight assigned by comparing March data to the actual 
model = ExponentialSmoothing (train['PMR']*1.023,trend = 'add',seasonal='add',seasonal_periods=12).fit()

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2019', periods=1000))

print(model.forecast(130))
model.forecast(130).to_csv('q2Result_final.csv')




