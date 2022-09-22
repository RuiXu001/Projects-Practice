# -*- coding: utf-8 -*-
"""
Created on Mon May 17 07:01:42 2021

@author: xurui
"""

import pandas as pd
#from statistics import mean
datadir = 'd:\msda\data298'
outputdir = 'd:\msda\data298'
df10 = pd.read_csv(datadir+'\modis\_81102.csv')
df10.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
df10 = df10[df10['Sample Measurement']>-0.0001]
df2 = pd.read_csv(datadir+'\modis\_88101.csv')
df2.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
df2 = df2[df2['Sample Measurement']>-0.0001]
df2
dfw = pd.read_csv(datadir+'\modis\_WIND.csv')
dfw.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
dfw = dfw[dfw['Sample Measurement']>-0.0001]
dft = pd.read_csv(datadir+'\modis\_TEMP.csv')
dft.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
dfp = pd.read_csv(datadir+'\modis\_PRESS.csv')
dfp.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
dfr = pd.read_csv(datadir+'\modis\_RH_DP.csv')
dfr.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
temp_df = df10.drop(columns=['Parameter Code','Parameter Name',
       'Time Local', 'Date GMT', 'Time GMT',
       'Sample Measurement'])
temp_df.drop_duplicates(inplace = True)
temp_df
df10m = temp_df
df10m['DailyPM10'] = 0
df10m['PeriodPM10'] = 0

df10m['wind'] = 0
df10m['temp'] = 0
df10m['press'] = 0
df10m['rhdp'] = 0
def mean(num):
    m = 0
    try:
        m = sum(num)/len(num)
        print(m, end=' ')

    except:
        pass
    return round(m,2)

def get_m_d():
    site_l = df10['Site Num'].unique()
    date_l = df10['Date Local'].unique()
    n= 0
    daily = []
    period = []
    for n in range(len(df10m)):
        print(n)
        #print(df10[(df10['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])])
        df10m.iloc[n, 5] = mean(df10[(df10['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])]['Sample Measurement'])#daily mean
        df10m.iloc[n, 6] = mean(df10[(df10['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])&(df10['Time GMT']< '20:00')&(df10['Time GMT']>'17:00')&(df10['Site Num']==df10m.iloc[n, 0])]['Sample Measurement'])#period

        df10m.iloc[n, 7] = mean(dfw[(dfw['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])&(dfw['Time GMT']< '20:00')&(dfw['Time GMT']>'17:00')&(dfw['Site Num']==df10m.iloc[n, 0])]['Sample Measurement'])#period
        df10m.iloc[n, 8] = mean(dft[(dft['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])&(dft['Time GMT']< '20:00')&(dft['Time GMT']>'17:00')&(dft['Site Num']==df10m.iloc[n, 0])]['Sample Measurement'])#period
        df10m.iloc[n, 9] = mean(dfp[(dfp['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])&(dfp['Time GMT']< '20:00')&(dfp['Time GMT']>'17:00')&(dfp['Site Num']==df10m.iloc[n, 0])]['Sample Measurement'])#period
        df10m.iloc[n, 10] = mean(dfr[(dfr['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])&(dfr['Time GMT']< '20:00')&(dfr['Time GMT']>'17:00')&(dfr['Site Num']==df10m.iloc[n, 0])]['Sample Measurement'])#period
get_m_d()
df10m
temp_df = df2.drop(columns=['Parameter Code','Parameter Name',
       'Time Local', 'Date GMT', 'Time GMT',
       'Sample Measurement'])
temp_df.drop_duplicates(inplace = True)
temp_df
df2m = temp_df
df2m['DailyPM25'] = 0
df2m['PeriodPM25'] = 0
df2m['wind'] = 0
df2m['temp'] = 0
df2m['press'] = 0
df2m['rhdp'] = 0
df2m

def get_m_d2():
    site_l = df2['Site Num'].unique()
    date_l = df2['Date Local'].unique()
    n= 0
    daily = []
    period = []
    for n in range(len(df2m)):
        print(n)
        df2m.iloc[n, 5] = mean(df2[(df2['Date Local'] == df2m.iloc[n, 3] )&(df2['Site Num']==df2m.iloc[n, 0])]['Sample Measurement'])#daily mean
        df2m.iloc[n, 6] = mean(df2[(df2['Date Local'] == df2m.iloc[n, 3] )&(df2['Site Num']==df2m.iloc[n, 0])&(df2['Time GMT']< '20:00')&(df2['Time GMT']>'17:00')&(df2['Site Num']==df2m.iloc[n, 0])]['Sample Measurement'])#period

        df2m.iloc[n, 7] = mean(dfw[(dfw['Date Local'] == df2m.iloc[n, 3] )&(df2['Site Num']==df2m.iloc[n, 0])&(dfw['Time GMT']< '20:00')&(dfw['Time GMT']>'17:00')&(dfw['Site Num']==df2m.iloc[n, 0])]['Sample Measurement'])#period
        df2m.iloc[n, 8] = mean(dft[(dft['Date Local'] == df2m.iloc[n, 3] )&(df2['Site Num']==df2m.iloc[n, 0])&(dft['Time GMT']< '20:00')&(dft['Time GMT']>'17:00')&(dft['Site Num']==df2m.iloc[n, 0])]['Sample Measurement'])#period
        df2m.iloc[n, 9] = mean(dfp[(dfp['Date Local'] == df2m.iloc[n, 3] )&(df2['Site Num']==df2m.iloc[n, 0])&(dfp['Time GMT']< '20:00')&(dfp['Time GMT']>'17:00')&(dfp['Site Num']==df2m.iloc[n, 0])]['Sample Measurement'])#period
        df2m.iloc[n, 10] = mean(dfr[(dfr['Date Local'] == df2m.iloc[n, 3] )&(df2['Site Num']==df2m.iloc[n, 0])&(dfr['Time GMT']< '20:00')&(dfr['Time GMT']>'17:00')&(dfr['Site Num']==df2m.iloc[n, 0])]['Sample Measurement'])#period

get_m_d2()
df2m
df10m.to_csv(r'd:\msda\data298\modis\averagepm10_.csv', index=False)
df2m.to_csv(r'd:\msda\data298\modis\averagepm2_.csv', index=False)
#%%%
def mean(num):
    m = 0
    try:
        m = sum(num)/len(num)
        print(m, end=' ')

    except:
        pass
    return round(m,2)

def get_m_d(df10, df10m):
    site_l = df10['Site Num'].unique()
    date_l = df10['Date Local'].unique()
    n= 0
    daily = []
    period = []
    for n in range(len(df10m)):
        print(n)
        #print(df10[(df10['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])])
        df10m.iloc[n, 5] = mean(df10[(df10['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])]['Sample Measurement'])#daily mean
        df10m.iloc[n, 6] = mean(df10[(df10['Date Local'] == df10m.iloc[n, 3] )&(df10['Site Num']==df10m.iloc[n, 0])&(df10['Time GMT']< '20:00')&(df10['Time GMT']>'17:00')&(df10['Site Num']==df10m.iloc[n, 0])]['Sample Measurement'])#period
    return df10m
#%%

datadir = 'd:\msda\data298'
outputdir = 'd:\msda\data298'
df2 = pd.read_csv(datadir+r'\modis\averagepm2.csv')
df10 = pd.read_csv(datadir+r'\modis\averagepm10.csv')
dfw = pd.read_csv(datadir+'\modis\_WIND.csv')
dfw.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
dfw = dfw[dfw['Sample Measurement']>-0.0001]
dft = pd.read_csv(datadir+'\modis\_TEMP.csv')
dft.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
dfp = pd.read_csv(datadir+'\modis\_PRESS.csv')
dfp.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)
dfr = pd.read_csv(datadir+'\modis\_RH_DP.csv')
dfr.drop(columns = ['Datum', 'POC', 'Unnamed: 0'], inplace = True)

del dfw
del dfw_m
del dfwm

dfw = dfr

temp_df = dfw.drop(columns=['Parameter Code','Parameter Name',
       'Time Local', 'Date GMT', 'Time GMT',
       'Sample Measurement'])
temp_df.drop_duplicates(inplace = True)
temp_df
dfwm = temp_df
dfwm['Dailydp'] = 0
dfwm['Perioddp'] = 0
dfw_m = get_m_d(dfw, dfwm)
dfwm.to_csv(r'd:\msda\data298\modis\average_dp.csv')
dfw_m

df2.columns
result = pd.merge(df2, dfwm, how="left", on=["Site Num", "Date Local"])
result
result.to_csv('')
