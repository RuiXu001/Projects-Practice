# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:16:16 2021

@author: xurui
"""

#%%
from datetime import datetime, timedelta
import numpy as np


import rasterio

import pandas as pd
import os
import glob

from affine import Affine
from pyproj import Proj, transform
import pyproj

from geopy.distance import geodesic

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pandas.io.json import json_normalize
from scipy.interpolate import griddata
import cv2
import json
import joblib
csv_output_dir = 'D:\xxx\xxx\epa\SaltonSea'
modis_path = r'd:\xxx\xxx\modis'
import pickle
import time
import requests
import tensorflow as tf
#%% basic funcs
def read_tif(img_dir):
    ds = rasterio.open(img_dir)
    T0 = ds.transform
    p1 = Proj(ds.crs)
    r = ds.read()[0] # so the shape will be [row, col], not [1, row, col]
    cols, rows = np.meshgrid(np.arange(r.shape[1]), np.arange(r.shape[0]))
    T1 = T0 * Affine.translation(0.5, 0.5)
    rc2en = lambda r, c: T1 * (c, r)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    return r, lats, longs
def get_rowcol(lats, longs, lat, long, n=3): 
    # get several (n=3) nearest point to the input lat&long
    # calculate the distance between input point and selected near point
    # return the nearest one
    lat_precision = 0.05
    lon_precision = 0.05
    l1 = []
    l2 = []
    while len(l1)<n or len(l2)<n: # not find enough lat or long
        lon_array = np.where((longs > long-lon_precision) & (longs < long+lon_precision))
        lat_array = np.where((lats > lat-lat_precision) & (lats < lat+lat_precision))

        l1 = np.unique([i for i in lon_array[0] if i in lat_array[0]])
        l2 = np.unique([i for i in lon_array[1] if i in lat_array[1]])
        if len(l1)<n:
            lat_precision += 0.01
        else:
            pass
        if len(l2)<n:
            lon_precision += 0.01
        else:
            pass
    dist = float('inf')
    for i in l1: #760
        for j in l2: #1539
            dist_ = geodesic((lats[i,j], longs[i,j]), (lat, long)).km
            if dist_ < dist:
                row, col = i, j
                dist = dist_
            else:
                pass
    return int(row), int(col)
def get_loc_relation(df, input_f): 
    # get the lats&longs from df and find the nearest one for every point
    # return as dict
    r, lats, longs = read_tif(input_f)
    dftmp = df[['Latitude', 'Longitude']].drop_duplicates().reset_index(drop = True)
    dftmp['row'] = np.nan
    dftmp['col'] = np.nan
    dict_rel_point = {}
    for i in range(len(dftmp)):
        lat = dftmp.loc[i, 'Latitude']
        long = dftmp.loc[i, 'Longitude']
        key = str(str(round(lat,3))+' '+str(round(long, 3)))
        row, col = get_rowcol(lats, longs, lat, long)
        dict_rel_point[key] = [row, col]
    return dict_rel_point
def fill(data):
    rows, cols = data.shape
    valid_rows, valid_cols = np.where(np.isnan(data) == False)
    valid_data = data[np.isnan(data) == False]
    for row in range(rows):
        for col in range(cols):
            min_ind = np.argmin(np.array([abs(row-i) for i in valid_rows])+np.array([abs(col-i) for i in valid_cols]))
            data[row, col] = valid_data[min_ind]
    return np.array(data)
def get3by3(matrix):
    '''[tl, top, tr,
    left, middle, right,
    bl, bottom, br]'''
    tl = matrix[:-2, :-2]
    top = matrix[0:-2, 1:-1]
    tr = matrix[:-2, 2:]
    left = matrix[1:-1, :-2]
    middle = matrix[1:-1, 1:-1]
    right = matrix[1:-1, 2:]
    bl = matrix[2:, :-2]
    bottom = matrix[2:, 1:-1]
    br = matrix[2:, 2:]
    return np.pad((tl+ top+ tr+ left+ middle+ right+ bl+ bottom+ br)/9, ((1,1), (1,1)), 'edge')


def get5by5(matrix):
    ''' 5 by 5: [m11, m12, m13, m14, m15, 
                 m21, m22, m23, m24, m25,
                 m31, m32, m33, m34, m35,
                 m41, m42, m43, m44, m45,
                 m51, m52, m53, m54, m55]'''
    m11 = matrix[:-4, :-4]
    m12 = matrix[:-4, 1:-3]
    m13 = matrix[:-4, 2:-2]
    m14 = matrix[:-4, 3:-1]
    m15 = matrix[:-4, 4:]
    m21 = matrix[1:-3, :-4]
    m22 = matrix[1:-3, 1:-3]
    m23 = matrix[1:-3, 2:-2]
    m24 = matrix[1:-3, 3:-1]
    m25 = matrix[1:-3, 4:]
    m31 = matrix[2:-2, :-4]
    m32 = matrix[2:-2, 1:-3]
    m33 = matrix[2:-2, 2:-2]
    m34 = matrix[2:-2, 3:-1]
    m35 = matrix[2:-2, 4:]
    m41 = matrix[3:-1, :-4]
    m42 = matrix[3:-1, 1:-3]
    m43 = matrix[3:-1, 2:-2]
    m44 = matrix[3:-1, 3:-1]
    m45 = matrix[3:-1, 4:]
    m51 = matrix[4:, :-4]
    m52 = matrix[4:, 1:-3]
    m53 = matrix[4:, 2:-2]
    m54 = matrix[4:, 3:-1]
    m55 = matrix[4:, 4:]
    return np.pad((m11+ m12+ m13+ m14+ m15+ m21+ m22+ m23+ m24+ m25
           + m31+ m32+ m33+ m34+ m35+ m41+ m42+ m43+ m44+ m45
           + m51+ m52+ m53+ m54+ m55)/25, ((2,2),(2,2)),'edge')
def score_model(y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'R2 score:  {r2_score(y_test, y_pred):.2f}, RMSE: {rmse:.2f}, MAE: {mae: .2f}')
    #print(f'range: {min(y_test)} to {max(y_test)}')
    return rmse, mae

def get_dist_ss(date_sf):
    f = open(f'D:\\xxx\\xxx\\Shapefiles\\SS{date_sf}.geojson')
    data = json.load(f)
    #print(f'D:\\xxx\\xxx\\Shapefiles\\SS{date_sf}.geojson loading')
    lats = np.load(r'd:\xxx\xxx\new\lats.npy')
    longs = np.load(r'd:\xxx\xxx\new\longs.npy')
    coors = data['features'][0]['geometry']['coordinates'][0]
    rowcol = []
    for i in coors:
        long,lat = i
        row, col = get_rowcol(lats, longs, lat, long)
        rowcol.append([row,col])
    empty_m = np.zeros((109,197))
    dist_to_ss = empty_m.copy()
    for r in range(empty_m.shape[0]):
        for c in range(empty_m.shape[1]):
            dist_min = np.min([((r-i[0])**2+(c-i[1])**2)**.5 for i in rowcol])
            dist_to_ss[r,c] = dist_min
    print(1)
    rowcol=np.array(rowcol)
    return dist_to_ss
def get_weather(date_obj, df_map):
    API_key = 'ba39aa444d9758071e834754b71da6b3'
    # minlat=28&maxlat=38&minlon=-120&maxlon=-110
    bbox = '-114,32,-119,37,10'
    url=f'http://api.openweathermap.org/data/2.5/box/city?bbox={bbox}&appid={API_key}'
    response = requests.get(url)
    resp = response.json()
    weather_data = json.dumps(resp['list'])
    df = pd.read_json(weather_data)
    df['lat'] = [i['Lat'] for i in df['coord']]
    df['long'] = [i['Lon'] for i in df['coord']]
    coor=[[df['lat'][i], df['long'][i]] for i in range(len(df))]
    unixtime = int(time.mktime(date_obj.timetuple()))
    w_dict = np.load(r'.\para&config\station_rowcol_w.npy',allow_pickle='TRUE').item()
    cols_w = ['TEMP', 'PRESS', 'RH_DP', 'WIND','Longitude','Latitude']
    dfw = pd.DataFrame()
    for i in range(len(coor)):
        lat, long = coor[i]
        url=f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={long}&dt={unixtime}&appid={API_key}&units=imperial'
        response = requests.get(url)
        resp = response.json()
        weather_data = json.dumps(resp['hourly'])
        df1 = pd.read_json(weather_data)
        df1.rename(columns={'wind_speed':'WIND', 'temp':'TEMP', 'pressure':'PRESS','humidity':'RH_DP'}, inplace=True)
        df1['Longitude']=long
        df1['Latitude']=lat
        
        col_d = dict()
        for col_w in cols_w:
            #print(round(np.mean(df1[col_w]),3))
            col_d[col_w]=round(np.mean(df1[col_w]),3)
        dfw=dfw.append(col_d,ignore_index=True)
    weather = ['TEMP', 'PRESS', 'RH_DP', 'WIND']

    for w in weather:
        rows=[]
        cols=[]
        data=np.zeros(lats2.shape)
        for i in range(len(dfw)):
            lat = dfw.loc[i, 'Latitude']
            long = dfw.loc[i, 'Longitude']
            key = str(str(round(lat,3))+' '+str(round(long, 3)))
            try:
                row,col = w_dict[key]
            except:
                print(key, 'key not found')
                row,col = get_rowcol(lats2, longs2, lat, long)
                w_dict[key]=[row,col]
                np.save(r'.\para&config\station_rowcol_w.npy', w_dict)
            rows.append(row)
            cols.append(col)
            data[row,col] = dfw.loc[i, w]
        points = np.array([[dfw.loc[i, 'Latitude'], dfw.loc[i, 'Longitude']] for i in range(len(dfw))])
        points = np.append(points,np.array([[lats2[579,693], longs2[579,693]]]),axis=0)
        values = dfw[w].values
        values = np.append(values, np.mean(values))
        np.append(points,[lats2[579,693], longs2[579,693]])
        grid_z1 = griddata(points, values, (lats2, longs2), method='cubic')#linear
        if np.isnan(grid_z1[470:579, 693:890]).sum()>0: # more than one nan. impute with mean value
            grid_z1 = np.where(np.isnan(grid_z1), np.mean(values), grid_z1)
        df_map[w] = np.round(np.reshape(grid_z1[470:579, 693:890], [109*197]),2)
    return df_map
#%% 


pm=10
date = '2021-12-15'
def pm_map(date, pm):
    print('start predicting')
    modis_path = r'd:\xxx\xxx\modis'
    csv_output_dir = 'D:\xxx\xxx\epa\Daily'
    files_047 = glob.glob(modis_path+'\\all_aod\\cropped\\*_047_*.tif')
    files_055 = glob.glob(modis_path+'\\all_aod\\cropped\\*_055_*.tif')
    lu_mcd12 = os.path.join(modis_path, 'mcd12\cropped')
    files_lu = glob.glob(lu_mcd12+'\*_LandUse_rep_cropped_.tif')
    NDVI_mod13 = os.path.join(modis_path,'mod13a2\cropped')
    lats2 = np.load(r'.\para&config\lats_whole.npy')
    longs2 = np.load(r'.\para&config\longs_whole.npy')
    ndvi = glob.glob(NDVI_mod13+'\*.tif')
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    year_day = date_obj.strftime('%Y%j')
    t1 = (date_obj-timedelta(days=31))
    t2 = (date_obj+timedelta(days=9))
    model_dir = f"d:\\New folder\\flask_demo\\model\\mlp_best_pm{str(pm)}.h5"
    datef = date_obj.strftime('%Y-%m-%d')
    year = date[:4]
    ''''''
    if pm == 10:
        order = ['year', 'AOD047_1', 'AOD055_1', 'AOD047_3',
           'AOD055_3', 'AOD047_5', 'AOD055_5', 'NDVI', 'WIND', 'TEMP', 'PRESS',
           'RH_DP', 'dist', 'month_10', 'month_11', 'month_12', 'month_2',
           'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
           'month_9', 'LU_36', 'LU_40', 'LU_9']
    elif pm ==2.5:
        order = ['AOD047_1', 'year', 'AOD055_1', 'AOD047_3',
        'AOD055_3', 'AOD047_5', 'AOD055_5', 'NDVI', 'WIND', 'TEMP', 'PRESS',
        'RH_DP', 'dist', 'month_10', 'month_11', 'month_12', 'month_2',
        'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8',
        'month_9', 'LU_30', 'LU_40', 'LU_9']
    
    if str(year_day)[:4] == '2020' or str(year_day)[:4] == '2021'or str(year_day)[:4] == '2022':
        tif_lu = [i for i in files_lu if '2019' == i.split('.h08v05.')[0][-7:-3]][0]
        #pm_dir = os.path.join(csv_output_dir, f'pm{pm}SS.csv')
        #dfpm = pd.read_csv(pm_dir)
    else:
        tif_lu = [i for i in files_lu if str(year_day)[:4] == i.split('.h08v05.')[0][-7:-3]][0]
        #pm_dir = os.path.join(csv_output_dir, f'pm{pm}SS.csv')
        #dfpm = pd.read_csv(pm_dir)
    tif_047 = [i for i in files_047 if str(year_day) == i.split('.h08v05.')[0][-7:]][0]
    tif_055 = [i for i in files_055 if str(year_day) == i.split('.h08v05.')[0][-7:]][0]
    tif_ndvi = [i for i in ndvi if (datetime.strptime(i.split('.h08v05.')[0][-7:], '%Y%j')>=t1) 
                and (datetime.strptime(i.split('.h08v05.')[0][-7:], '%Y%j')<t2)][0]
    
    aod047 = rasterio.open(tif_047).read()[0]
    aod055 = rasterio.open(tif_055).read()[0]
    lats1 = np.load(r'.\para&config\lats.npy')
    longs1 = np.load(r'.\para&config\longs.npy')
    rlu = rasterio.open(tif_lu).read()[0]
    r_lu = cv2.resize(rlu, (197,109))
    rn = rasterio.open(tif_ndvi).read()[0]
    
    # create the dataframe 109*197 21473 records
    df_map = pd.DataFrame()
    df_map['Latitude']=np.reshape(lats1, [109*197])
    df_map['Longitude']=np.reshape(longs1, [109*197])
    #df_map['year'] = str(date[:4])
    df_map['LU'] = np.reshape(r_lu, [109*197])
    df_map['NDVI'] = np.reshape(rn, [109*197])
    df_map['month'] = str(date.split('-')[1])
    df_map['AOD047_1'] = np.reshape(aod047, [109*197])
    df_map['AOD055_1'] = np.reshape(aod055, [109*197])
    df_map.loc[(df_map['AOD047_1']<-100), 'AOD047_1']=0
    df_map.loc[(df_map['AOD055_1']<-100), 'AOD055_1']=0
    '''
    # average weather
    weather_vari = ['WIND', 'TEMP', 'PRESS', 'RH_DP']
    for vari in weather_vari:
        df_map[vari] = np.mean(dfpm.loc[dfpm['Date Local']==date][vari])
    '''
    # cubic interpolate weather data
    # weather data from epa are historical data. did not provide realtime or recent data. use open weather instead (get_weather(date_obj, df_map)).
    weather_vari = ['WIND', 'TEMP', 'PRESS', 'RH_DP']
    w_para = {'TEMP':62101, 'WIND':61103, 'PRESS':64101, 'RH_DP':62201}
    
    req_date = date_obj.strftime('%Y%m%d')
    w_dict = np.load(r'.\para&config\station_rowcol_w.npy',allow_pickle='TRUE').item()
    try:
        for vari in weather_vari:
            response = requests.get(f'https://aqs.epa.gov/data/api/dailyData/byBox?email=rui.xu01@sjsu.edu&key=orangegazelle43&param={w_para[vari]}&bdate={req_date}&edate={req_date}&minlat=28&maxlat=38&minlon=-120&maxlon=-110')
            resp = response.json()
            weather_data = json.dumps(resp['Data'])
            df = pd.read_json(weather_data) # convert requested data into df.
            rows=[]
            cols=[]
            data=np.zeros(lats2.shape)
            for i in range(len(df)):
                lat = df.loc[i, 'latitude']
                long = df.loc[i, 'longitude']
                key = str(str(round(lat,4))+' '+str(round(long, 4)))
                try:
                    row,col = w_dict[key]
                except:
                    print(key, 'key not found')
                    row,col = get_rowcol(lats2, longs2, lat, long)
                    w_dict[key]=[row,col]
                    np.save(r'.\para&config\station_rowcol_w.npy', w_dict)
                rows.append(row)
                cols.append(col)
                data[row,col] = df.loc[i, 'arithmetic_mean']
            points = np.array([[df.loc[i, 'latitude'], df.loc[i, 'longitude']] for i in range(len(df))])
            points = np.append(points,np.array([[lats2[579,693], longs2[579,693]]]),axis=0)
            values = df['arithmetic_mean'].values
            values = np.append(values, np.mean(values))
            np.append(points,[lats2[579,693], longs2[579,693]])
            grid_z1 = griddata(points, values, (lats2, longs2), method='cubic')
            if np.isnan(grid_z1[470:579, 693:890]).sum()>0: # more than one nan. impute with mean value
                grid_z1 = np.where(np.isnan(grid_z1), np.mean(values), grid_z1)
            df_map[vari] = np.round(np.reshape(grid_z1[470:579, 693:890], [109*197]),2)
    except:
        df_map = get_weather(date_obj, df_map)
    # average of 3by3 and average of 5by5.
    aod047_3 = np.reshape(get3by3(aod047), [109*197])
    aod055_3 = np.reshape(get3by3(aod055), [109*197])
    aod047_5 = np.reshape(get5by5(aod047), [109*197])
    aod055_5 = np.reshape(get5by5(aod055), [109*197])
    df_map['AOD047_3'] = np.where(aod047_3<0, 0, aod047_3)
    df_map['AOD055_3'] = np.where(aod055_3<0, 0, aod055_3)
    df_map['AOD047_5'] = np.where(aod047_5<0, 0, aod047_5)
    df_map['AOD055_5'] = np.where(aod055_5<0, 0, aod055_5)
    sfs = glob.glob('D:\\xxx\\xxx\\Shapefiles\\*.geojson')   
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    date_ = date_obj.strftime('%Y%m%d')
    date_sf = [str(i.split('SS')[1][:8]) for i in sfs]
    date_sf_d = [abs(datetime.strptime(d, '%Y%m%d')-date_obj) for d in date_sf]
    date_of_sf = date_sf[np.argmin(date_sf_d)]
    dist_m = get_dist_ss(date_of_sf)
    
    df_map['dist'] = np.reshape(dist_m, [109*197])
    
    # dummy and standardlization
    df_map['month'] = df_map['month'].apply(str)
    df_map['LU'] = df_map['LU'].apply(str)
    df=pd.read_csv(r'D:/New folder/flask_demo/para&config/map_county&zipcode.csv')
    df_map['county']=df['county']
    df_map['zipcode']=df['zipcode']
    df_map_ori = df_map.copy
    #df_map=df_map[(df_map['county']=='Riverside County')|(df_map['county']=='Imperial County')]
    
    
    df_map_pro = df_map.drop(columns = ['Latitude', 'Longitude'])
    df_map_pro = pd.get_dummies(data=df_map_pro, drop_first=True)
    
    for i in order:
        if i not in list(df_map_pro.columns):
            df_map_pro[i]=0
    
    df_map_dum_ord = df_map_pro[order] # for the land use may have more dummy vari which did not show in the training, remove them
    sc_X = pickle.load(open(f'.\\para&config\\scaler_pm{pm}.pkl', 'rb'))
    X=sc_X.fit_transform(df_map_dum_ord)
    
    # load the trained model
    #saved_model = joblib.load(model_dir)
    saved_model = tf. keras. models. load_model(model_dir)
    #saved_model2 = joblib.load(model_dir2)
    #saved_model3 = joblib.load(model_dir3)
    
    #results = saved_model.predict(X)
    results = saved_model.predict(X)
    #results2 = saved_model2.predict(X)
    #results3 = saved_model3.predict(X)
    df_map['PM_concentration'] = np.round(np.array(results))
    
    #df_map['Pred_MLP'] = np.array(results1)
    #df_map['Pred_RFR'] = np.array(results2)
    #df_map['Pred_SVR'] = np.array(results3)
    return df_map[['Latitude', 'Longitude','county', 'zipcode', 'PM_concentration']]

#df = pm_map('2013-02-01', 2.5)




#%%
date = '2022-01-01'
date_obj = datetime.strptime(date, '%Y-%m-%d')

for i in range(13):
    date = datetime.strftime(date_obj+timedelta(days=i), '%Y-%m-%d')
    output_f1 = f'./pred/pm2.5_{date}.csv'
    if os.path.exists(output_f1):
        print(output_f1, 'exists')
    else:
        year = date[:4]
        print(date, year, i, end=' ')
        dftmp1 = pm_map(date, pm=2.5)
        time.sleep(3)
        dftmp1['Date Local'] = date
        dftmp1.to_csv(f'./pred/pm2.5_{date}.csv', index=False)

        dftmp2 = pm_map(date, pm=10)
        time.sleep(3)
        dftmp2['Date Local'] = date
        dftmp2.to_csv(f'./pred/pm10_{date}.csv', index=False)
        print('Saved ', date, [2.5,10])


#%%
from boto3.session import Session
import time
import pandas as pd
import numpy as np
import keplergl
from keplergl import KeplerGl
AWSAccessKeyId=''
AWSSecretKey=''
session = Session(aws_access_key_id=AWSAccessKeyId, aws_secret_access_key=AWSSecretKey, region_name='us-west-2')
s3 = session.client("s3")
pre1='results/pm10'
response = s3.list_objects_v2(Bucket='',Prefix=pre1)
dfsum = pd.DataFrame()
for key in sorted([response['Contents'][-i]['Key'] for i in range(7)], reverse=True):
    df = pd.read_csv(f'https://xxx.s3.us-west-2.amazonaws.com/{key}')
    t = time.mktime(time.strptime(df['Date Local'][0], '%Y-%m-%d'))+43200
    print(key, df['Date Local'][0], t)
    df['time'] = t
    df1 = df[(df['county']=='Riverside County')|
            (df['county']=='Imperial County')|
            (df['county']=='San Diego County')]
    dfsum = pd.concat([dfsum, df1], ignore_index = True)
dfsum=dfsum[['Latitude', 'Longitude', 'county', 'zipcode', 'PM_concentration','time']]
# dfsum.to_csv('pm10.csv',index=False)
map1 = KeplerGl()
map1.add_data(data=dfsum, name='pred')

true=True
false=False
null=''
pm='PM10'
config={"version":"v1","config":{"visState":{"filters":[{"dataId":["pred"],"id":"14bgjfetn","name":["time"],"type":"timeRange","value":[1630886583000,1630973157000],"enlarged":true,"plotType":"histogram","animationWindow":"free","yAxis":null,"speed":5}],"layers":[{"id":"bjslvu","type":"grid","config":{"dataId":"pred","label":pm,"color":[255,153,31],"highlightColor":[252,242,26,255],"columns":{"lat":"Latitude","lng":"Longitude"},"isVisible":true,"visConfig":{"opacity":0.2,"worldUnitSize":2,"colorRange":{"name":"Uber Viz Sequential 4","type":"sequential","category":"Uber","colors":["#E6FAFA","#C1E5E6","#9DD0D4","#75BBC1","#4BA7AF","#00939C"]},"coverage":1,"sizeRange":[0,500],"percentile":[0,100],"elevationPercentile":[0,100],"elevationScale":2.6,"enableElevationZoomFactor":true,"colorAggregation":"average","sizeAggregation":"count","enable3d":false},"hidden":false,"textLabel":[{"field":null,"color":[255,255,255],"size":18,"offset":[0,0],"anchor":"start","alignment":"center"}]},"visualChannels":{"colorField":{"name":"PM_concentration","type":"real"},"colorScale":"quantile","sizeField":null,"sizeScale":"linear"}}],"interactionConfig":{"tooltip":{"fieldsToShow":{"tai4l791h":[{"name":"0","format":null},{"name":"Latitude","format":null},{"name":"Longitude","format":null},{"name":"county","format":null},{"name":"zipcode","format":null}]},"compareMode":false,"compareType":"absolute","enabled":true},"brush":{"size":0.5,"enabled":false},"geocoder":{"enabled":true},"coordinate":{"enabled":false}},"layerBlending":"normal","splitMaps":[],"animationConfig":{"currentTime":null,"speed":1}},"mapState":{"bearing":0,"dragRotate":false,"latitude":33.34268094529604,"longitude":-115.96873911943312,"pitch":0,"zoom":9.633763175836808,"isSplit":false},"mapStyle":{"styleType":"satellite","topLayerGroups":{"label":true},"visibleLayerGroups":{},"threeDBuildingColor":[3.7245996603793508,6.518049405663864,13.036098811327728],"mapStyles":{}}}}
#config = {"version":"v1","config":{"visState":{"filters":[{"dataId":["pred"],"id":"o4287fjc","name":["time"],"type":"timeRange","value":[1639076400000,1639598400000],"enlarged":true,"plotType":"histogram","animationWindow":"free","yAxis":null,"speed":1}],"layers":[{"id":"jpff04o","type":"grid","config":{"dataId":"pred","label":pm,"color":[255,203,153],"highlightColor":[252,242,26,255],"columns":{"lat":"Latitude","lng":"Longitude"},"isVisible":true,"visConfig":{"opacity":0.1,"worldUnitSize":2,"colorRange":{"name":"Uber Viz Sequential 4","type":"sequential","category":"Uber","colors":["#E6FAFA","#C1E5E6","#9DD0D4","#75BBC1","#4BA7AF","#00939C"]},"coverage":1,"sizeRange":[0,500],"percentile":[0,100],"elevationPercentile":[0,100],"elevationScale":5,"enableElevationZoomFactor":true,"colorAggregation":"average","sizeAggregation":"count","enable3d":false},"hidden":false,"textLabel":[{"field":null,"color":[255,255,255],"size":18,"offset":[0,0],"anchor":"start","alignment":"center"}]},"visualChannels":{"colorField":{"name":"PM_concentration","type":"real"},"colorScale":"quantile","sizeField":null,"sizeScale":"linear"}}],"interactionConfig":{"tooltip":{"fieldsToShow":{"mvngs185":[{"name":"Latitude","format":null},{"name":"Longitude","format":null}]},"compareMode":false,"compareType":"absolute","enabled":true},"brush":{"size":0.5,"enabled":false},"geocoder":{"enabled":true},"coordinate":{"enabled":false}},"layerBlending":"normal","splitMaps":[],"animationConfig":{"currentTime":null,"speed":1}},"mapState":{"bearing":0,"dragRotate":false,"latitude":33.365100030242644,"longitude":-115.86661842495816,"pitch":0,"zoom":11.267526351673617,"isSplit":false},"mapStyle":{"styleType":"dark","topLayerGroups":{"label":true,"water":false},"visibleLayerGroups":{"label":true,"road":true,"border":false,"building":true,"water":true,"land":true,"3d building":false},"threeDBuildingColor":[9.665468314072013,17.18305478057247,31.1442867897876],"mapStyles":{}}}}
map1.config = config
map1.save_to_html(file_name=f'{pm}.html',read_only=False)

#%%
dfsum['Date Local'].drop_duplicates()

df['Date Local'] = '2021-12-16'

dfsum.drop_duplicates(inplace=True)


