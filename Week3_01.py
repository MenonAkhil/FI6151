# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:41:50 2019

@author: Finbarr.Murphy
"""

import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

plt.style.use('ggplot')
start = dt.datetime(2016,1,1)
end = dt.datetime(2019,2,28)

myStocks = ['AAPL','FB','PYPL']

df = web.DataReader(myStocks,'yahoo',start,end)

df = pd.DataFrame(df['Adj Close'].pct_change())
df = df.iloc[1:] # delete first row

ax1 = df.plot.scatter(x='AAPL', y='FB', c='DarkBlue')

# plot
plt.figure(figsize=(8,8))
sns.regplot(x=df['AAPL'], y=df['FB'], line_kws={'color':'b','alpha':0.7,'lw':5})
sns.jointplot('AAPL', 'FB', df, kind='scatter', color='seagreen')

df.corr()


import statsmodels.api as sm

X = df['AAPL']
y = df['FB']

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
