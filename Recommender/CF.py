# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 19:51:56 2022

@author: xurui
"""

# Item-based collaborative filtering
# User-based collaborative filtering

#%% import libaries
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns


#%% datasets
base_dir='d:/kaggle/book recommendation system'

os.chdir(base_dir)
dfbooks=pd.read_csv('Books.csv')
dfratings=pd.read_csv('Ratings.csv')
dfusers=pd.read_csv('Users.csv')

print(dfbooks.info())
print(dfratings.info())
print(dfusers.info())

df1 = dfbooks.merge(dfratings, how='left', on='ISBN')
df2 = df1.merge(dfusers, how='left', on='User-ID')

df=df2.copy()
df.head()
df.info()
df=df[['ISBN','Book-Title','Book-Author','User-ID','Book-Rating']]
df.isna().sum()
#%% data preprocessing
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.isna().sum()
#df['Age']=df['Age'].astype('int')
df['User-ID']=df['User-ID'].astype('int')
df['Book-Rating']=df['Book-Rating'].astype('int')

print('Count on users: {:,}'.format (df['User-ID'].nunique()))
print('Count on items (books): {:,}'.format (df['Book-Title'].nunique()))

# remove rarely rating users and rarely read books
user_c=df['User-ID'].value_counts()
book_c=df['Book-Title'].value_counts()
dfu_count=pd.DataFrame({'User-ID':user_c.index, 'count':user_c.values})
dfb_count=pd.DataFrame({'Book-Title':book_c.index, 'count':book_c.values})

#n_read=np.array(df['Book-Title'].value_counts()) #number of users read the book 
#values, frequencies=np.unique(n_read,return_counts=True)
dfu_count.sort_values(by='count', ascending=False,inplace=True)
dfb_count.sort_values(by='count', ascending=False,inplace=True)
#len(dfb_count[dfb_count['count']<30])/(df['Book-Title'].nunique())*100
#sum(dfb_count[dfb_count['count']>=30]['count'])
common=df[(df['User-ID'].isin(
    dfu_count[dfu_count['count']>=500]['User-ID']))&
    (df['Book-Title'].isin(
    dfb_count[dfb_count['count']>=100]['Book-Title']))
    ].reset_index(drop=True)
user_item_rating=common.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")
print('{0:,} users, {1:,} items'.format(user_item_rating.shape[0], user_item_rating.shape[1]))
user_item_rating.fillna(0, inplace=True)
user_item_rating.iloc[:5,:5]
sns.heatmap(user_item_rating)
#501 * 2444-user_item_rating.isna().sum().sum()
#len(common)-common['Book-Rating'].isna().sum()
#%% book/item-based recommendation system
# try 1
#bookname=common_book.sample()['Book-Title'].iloc[0]# Randomly select a book
user_item_rating=common.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating")
user_item_rating.fillna(0, inplace=True)
bookname='The Da Vinci Code'
rating=user_item_rating[bookname]
sum([i>0 for i in rating])
rating.sort_values(ascending=False)
user_item_rating.corrwith(rating).sort_values(ascending=False).head(10)
#########################################
# user-based recommendation 
# try 2 using knn find the n nearest users 
from sklearn.neighbors import NearestNeighbors
user_item_rating=common.pivot_table(columns=["User-ID"],index =["Book-Title"], values="Book-Rating")
user_item_rating.fillna(0, inplace=True)
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_rating.values)
distances, indices = knn.kneighbors(user_item_rating.values, n_neighbors=5)

ind_item=user_item_rating.index.tolist().index('The Da Vinci Code')
sim_item=indices[ind_item].tolist()
item_distances=distances[ind_item].tolist()
id_item=sim_item.index(ind_item)
for i,j in zip(sim_item[1:], item_distances[1:]):
    print('Item: {0}  | Distance: {1:.2f}'.format(user_item_rating.index[i],j))

#%%



