#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Finance
# 
# ```
# Created By: Ravi Kasarla
# Creation Date: 02-SEP-2019
# Last Updated: 08-SEP-2019
# Base source: https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7
# ```

# In[646]:


import pandas as pd
import datetime
from pandas import Series, DataFrame


# ### Get stock data from Yahoo Finance
# ```
# URL: https://finance.yahoo.com/quote/AAPL/history?p=AAPL
# Make sure to adjust data range, apply and then download the file
# and save it in data directory under the current folder
# ```

# In[647]:


df = pd.read_csv('data/AAPL.csv', header=0, index_col='Date', parse_dates=True)


# In[648]:


df.head()


# In[649]:


df.tail()


# ### Moving Average

# In[650]:


close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()


# In[651]:


mavg.tail()


# In[652]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend();


# ### Return Deviation — to determine risk and return
# ```
# Expected Return measures the mean, or expected value, of the probability distribution of investment returns. The expected return of a portfolio is calculated by multiplying the weight of each asset by its expected return and adding the values for each investment — Investopedia.
# 
# Following is the formula you could refer to:
# rt = (pt / (pt-1)) - 1
# ```

# In[653]:


rets = close_px / close_px.shift(1) - 1
rets.plot(label='return');


# ### Predicting Stocks Price
# ```
# We will be using the following models:
# Simple Linear Analysis, Quadratic Discriminant Analysis (QDA), and K Nearest Neighbor (KNN)
# ```

# In[654]:


dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0


# In[655]:


dfreg.head()


# In[656]:


import math
import numpy as np
import sklearn.preprocessing as preprocessing


# In[657]:


# Example Dealing with NaN
d = {'filename': ['M66_MI_NSRh35d32kpoints.dat', 'F71_sMI_DMRI51d.dat', 'F62_sMI_St22d7.dat', 'F41_Car_HOC498d.dat', 'F78_MI_547d.dat'], 'alpha1': [0.8016, 0.0, 1.721, 1.167, 1.897], 'alpha2': [0.9283, 0.0, 3.833, 2.809, 5.459], 'gamma1': [1.0, np.nan, 0.23748000000000002, 0.36419, 0.095319], 'gamma2': [0.074804, 0.0, 0.15, 0.3, np.nan], 'chi2min': [39.855990000000006, 1e+25, 10.91832, 7.966335000000001, 25.93468]}
dfe = pd.DataFrame(d).set_index('filename')


# In[658]:


dfe


# In[659]:


# If there are any NaN get that into a new DF
dfenan = dfe[dfe.isna().any(axis=1)]


# In[660]:


dfenan


# In[661]:


# Fill NaN with 
dfe = dfe.fillna(dfe.mean())


# In[662]:


dfe


# In[ ]:





# In[ ]:





# In[663]:


# Check if there are any NaN
dfreg[dfreg.isna().any(axis=1)]


# In[664]:


# In this data set there are no NaNs, just in case you have NaN's replace with Average
dfreg.fillna(value=dfreg.mean(), inplace=True)


# In[665]:


# Drop missing value
# dfreg.fillna(value=-99999, inplace=True)


# In[666]:


# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
forecast_out


# In[667]:


dfreg.head()


# In[668]:


# Separating the label here, we want to predict the AdjClose
# Note you have shifted Label with 28, so your label has 28 days future value
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)


# In[669]:


dfreg.head(2)


# In[670]:


dfreg.tail(2)


# In[671]:


dfreg.shape


# In[672]:


dfregNoy = dfreg.drop(['label'],1)


# In[673]:


dfregNoy.shape


# In[674]:


dfregNoy.head(2)


# In[675]:


X = dfregNoy


# In[676]:


X.shape


# In[677]:


# First two rows -- if numpy
# X[:2,:]


# In[678]:


# Last two rows -- if numpy
# X[-2:,:]


# In[679]:


pd.options.display.float_format = '{:.2f}'.format


# In[680]:


dfreg.describe().T


# In[681]:


X = np.array(dfreg.drop(['label'], 1))


# In[682]:


X = preprocessing.scale(X)


# In[683]:


# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


# In[684]:


X.shape


# In[685]:


X_lately.shape


# In[686]:


# Separate label and identify it as y
y = dfreg['label']
y = y[:-forecast_out]


# In[687]:


y.shape


# In[688]:


from sklearn.model_selection import train_test_split


# In[689]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[690]:


from sklearn.preprocessing import MinMaxScaler


# In[691]:


sclar = MinMaxScaler()


# In[692]:


# X_train = pd.DataFrame(data=scaler.fit_transform(X_train),columns = X_train.columns,index=X_train.index)
X_train = pd.DataFrame(data=scaler.fit_transform(X_train))


# In[693]:


# X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)
X_test = pd.DataFrame(data=scaler.transform(X_test))


# ### Model Generation

# In[694]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[695]:


# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)


# In[696]:


# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)


# In[697]:


# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)


# In[ ]:





# In[698]:


# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


# In[699]:


confidencereg = clfreg.score(X_test, y_test)
confidencereg


# In[700]:


confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly2


# In[701]:


confidencepoly3 = clfpoly3.score(X_test,y_test)
confidencepoly3


# In[702]:


confidenceknn = clfknn.score(X_test, y_test)
confidenceknn


# In[703]:


# Predict Y using linear regression
forecast_set = clfreg.predict(X_lately)


# In[704]:


forecast_set


# In[705]:


np.set_printoptions(formatter={'float_kind':'{:0.2f}'.format})


# In[706]:


forecast_set


# In[707]:


dfreg['Forecast'] = np.nan


# In[708]:


dfreg.head()


# In[709]:


last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[ ]:





# In[ ]:




