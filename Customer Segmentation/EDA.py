# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 00:46:48 2021

@author: xurui
"""

import os
import math
import scipy
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import randint
from scipy.stats import loguniform
from IPython.display import display

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

base_dir=r'D:\kaggle\customer segmentation'
os.chdir(base_dir)
df_org = pd.read_excel('Online Retail.xlsx')
df=df_org.copy()
df.head()
df.columns
print(df.isna().sum())
df.info()
df.describe()
# df=df.sample(20000, random_state=0)


# Follow MFM model
# for each customer, get the most recent inovice date as the recency
#                        the number of buying as frequency, 
#                        the total sum as monetary value.
df['TotalSum']=df['Quantity']*df['UnitPrice']
df['InvoiceDate'] = df['InvoiceDate'].dt.date # date time to date
latest_day = max(df['InvoiceDate'] ) + datetime.timedelta(days=1)

df_customers = df.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (latest_day-x.max()).days,
    'InvoiceNo':'count',
    'TotalSum':'sum'})



df_customers.rename(columns={'InvoiceDate': 'Recency',
                             'InvoiceNo': 'Frequency',
                             'TotalSum': 'MonetaryValue'}, inplace=True)

df_customers.info()
df_customers.describe()
df_customers.isnull().sum()


#%% EDA
'''
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
f.set_figwidth(15)
ax1.hist(df_customers.Recency)
ax2.hist(df_customers.Frequency)
ax3.hist(df_customers.MonetaryValue)'''

def show_dist(df):
    cols = [i for i in df.columns]
    plt.figure(figsize = [15, 10])
    for c in range(len(cols)):
        plt.subplot(3,3,c+1)
        sns.distplot(df[cols[c]])
    plt.tight_layout()
    plt.show()
    return

def show_qqplots(df, title=''):
    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(1, 3, 1)
    qqplot(df[cols[0]], line='45', ax=ax1)
    ax1.title.set_text('{0} - {1}'.format(title, cols[0]))
    ax2 = fig.add_subplot(1, 3, 2)
    qqplot(df[cols[1]], line='45',ax=ax2)
    ax2.title.set_text('{0} - {1}'.format(title, cols[1]))
    ax3 = fig.add_subplot(1, 3, 3)
    qqplot(df[cols[2]], line='45',ax=ax3)
    ax3.title.set_text('{0} - {1}'.format(title, cols[2]))
    fig.tight_layout()
    plt.gcf()
    return 

show_dist(df_customers)
show_qqplots(df_customers)

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot

sm.qqplot(df_customers['Recency'], line='45')
stats.probplot(df_customers['Frequency'], )

qqplot(df_customers['MonetaryValue'], line='45')
plt.show()

# variables did not follow normal distribution. transform the data
cutomers_logT = df_customers.copy(deep=True) # logist  
cutomers_sqrtT = df_customers.copy(deep=True) # square root
cutomers_cbrtT = df_customers.copy(deep=True) # cube-root 
cutomers_bxcxT = df_customers.copy(deep=True) # Box-Cox power transformation

for c in cols:
    cutomers_logT[c] = np.log(df_customers[c]).apply(lambda x: np.nan if x == float('-inf') else x)
    cutomers_sqrtT[c] = np.sqrt(df_customers[c])
    cutomers_cbrtT[c] = np.cbrt(df_customers[c])
    if c!=cols[2]: # Monetary Value contains neg values
    cutomers_bxcxT[c] = stats.boxcox(df_customers[c])


show_dist(cutomers_logT)
show_dist(cutomers_sqrtT)
show_dist(cutomers_cbrtT)
show_qqplots(cutomers_logT,'log')
show_qqplots(cutomers_sqrtT,'sqrt')
show_qqplots(cutomers_cbrtT,'cbrt')
