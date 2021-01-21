# Q1


#Model Based Approach 

# Business Problem : Forecasting the spotify music data for the next 5 years.
# Business Objective : Maximize sales/profits

# Import the libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

%matplotlib inline

import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import svm
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("I:\Downloads(IDM) & Browser\spotify_data\\data.csv")

time_series = df[['tempo','release_date']]

time_series.isnull().sum()

time_series = time_series[['release_date','tempo']]

plt.figure(figsize=(40,20))
plt.title("Song Trends Over Time", fontdict={"fontsize": 15})

lines = ["tempo"]

for line in lines:
    ax = sns.lineplot(x='release_date', y=line, data= time_series)
plt.ylabel("value")
plt.legend(lines)

#Also, a given time series is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise.

#These components are defined as follows:

#Level: The average value in the series.
#Trend: The increasing or decreasing value in the series.
#Seasonality: The repeating short-term cycle in the series.
#Noise: The random variation in the series.
#First, we need to check if a series is stationary or not because time series analysis only works with stationary data.





# Check for missing values
df.isnull().sum()

# Drop unneccessary columns
df.drop(["id", "key", "mode", "explicit", "release_date"], axis=1, inplace=True)

corr = df[["acousticness","danceability","energy", "instrumentalness", 
           "liveness","tempo", "valence", "loudness", "speechiness"]].corr()

plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True)


#There is a strong positive correlation between energy and loudness as we suspected. On the other hand, there seems to be a strong negative correlation between energy and acousticness.


#The dataset contains songs from as far back as 1921. We can get an overview how the characteristics of song change over a hundred-year-period.

year_avg = df[["acousticness","danceability","energy", "instrumentalness", 
               "liveness","tempo", "valence", "loudness", "speechiness","year"]].\
groupby("year").mean().sort_values(by="year").reset_index()

year_avg.head()



# Create a line plot
plt.figure(figsize=(18,8))
plt.title("Song Trends Over Time", fontdict={"fontsize": 15})

lines = ["acousticness","danceability","energy", 
         "instrumentalness", "liveness", "valence", "speechiness"]

for line in lines:
    ax = sns.lineplot(x='year', y=line, data=year_avg)
    
    
plt.ylabel("value")
plt.legend(lines)

df["t"] = np.arange(1,169910)
df["t_squared"] = df["t"]*df["t"]



df["log_duration_ms"] = np.log(df["duration_ms"])



year_dummies = pd.DataFrame(pd.get_dummies(df['year']))
df1 = pd.concat([df, year_dummies], axis = 1)

df1.duration_ms.plot()

Train = df1.tail(150000)
Test = df1.head(19909)

####################### L I N E A R ##########################

import statsmodels.formula.api as smf 

linear_model = smf.ols('duration_ms ~ t', data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['duration_ms']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_duration_ms ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['duration_ms']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('duration_ms ~ t+t_squared', data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['duration_ms'])-np.array(pred_Quad))**2))
rmse_Quad






tsa_plots.plot_acf(df.tempo, lags = 12)


model1 = ARIMA(df.tempo, order = (1,1,6)).fit(disp=0)
model2 = ARIMA(df.tempo, order = (1,1,5)).fit(disp=0)
model1.aic
model2.aic

p=0
q=0
d=2
pdq=[]
aic=[]
for q in range(7):
    try:
        model = ARIMA(df.tempo, order = (p, d, q)).fit(disp = 0)
        x=model.aic
        x1= p,d,q
        aic.append(x)
        pdq.append(x1)
    except:
        pass
            
keys = pdq
values = aic
d = dict(zip(keys, values))
print (d)




