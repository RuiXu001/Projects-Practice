# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:42:33 2021

@author: xurui
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import waterdetect as wd
import rasterio
# import geopandas as gpd
from osgeo import gdal, gdal_array

import os
wdir = 'd:\xxx\xxx\salton sea'
outputpath = 'd:\xxx\xxx\salton sea\wd'
inputpath = 'e:\SaltonSea039037'
os.chdir(wdir)
os.getcwd()

#%%
def get_path(date):
    path = inputpath + '\LC08_L1TP_039037_'+date+'_01_T1' 
    #print(path)
    return path
#get_path('2012123')
def get_band(date, band):
    path = get_path(date) + '\LC08_L1TP_039037_'+date+'_01_T1_B'+str(band)+'.TIF'
    #print(path)
    return path
#print(get_band('20130214-20170101', 3))


#%% TRY
with open('../Salton_039037.txt') as f:
    listf = f.readlines()
list_d = [i.replace('\n','') for i in listf]
print(list_d)
config = wd.DWConfig(config_file='d:\xxx\xxx\WaterDetect.ini')


date1 = list_d[0]
b3path = get_band(date1, 3)
b5path = get_band(date1, 5)
with rasterio.open(b3path) as src3:
    b3 = src3.read()
    profile = src3.profile.copy()
with rasterio.open(b5path) as src5:
    nir = src5.read()
b3.shape
bands = {'Green': b3.squeeze()/10000, 'Nir': nir.squeeze()/10000}
wmask = wd.DWImageClustering(bands=bands, bands_keys=['Nir', 'ndwi'], invalid_mask=None, config=config)
mask = wmask.run_detect_water()
plt.imshow(wmask.cluster_matrix)
plt.imshow(wmask.water_mask==1)
type(wmask.cluster_matrix)

print(wmask.cluster_matrix)
wmask.cluster_matrix.shape
profile.update({
          'dtype':'float64',
          'height': wmask.cluster_matrix.shape[0],
          'width':wmask.cluster_matrix.shape[1],

      })
print(profile)
fn = outputpath+b3path[-38:-6]+b3path[-4:]
print(fn)
with rasterio.open(fn, 'w', **profile) as dst:
    dst.write_band(1, wmask.cluster_matrix)

shp_clip = 'd:\xxx\xxx\salton sea\SaltonSeaArea.shp'
band = b3path
options = gdal.WarpOptions(cutlineDSName=shp_clip,cropToCutline=True)
outBand = gdal.Warp(srcDSOrSrcDSTab= band,
                    destNameOrDestDS=band[:-4]+'_c2'+band[-4:],
                    options=options)

with rasterio.open(fn,'r') as src:
    ss = src.read()
    print(src.profile)
ss[0]
plt.imshow(ss[0])
#%%

with open('../Salton_039037.txt') as f:
    listf = f.readlines()
list_d = [i.replace('\n','') for i in listf]
print(list_d)
config = wd.DWConfig(config_file='d:\xxx\xxx\WaterDetect.ini')

print(list_d[98])
fpath = 'http://landsat-pds.s3.amazonaws.com/c1/L8/'+path+'/'+row+'/LC08_L1TP_'+path+row+'_'+Acquisition_Processing+'_01_T1/LC08_L1TP_'+path+row+'_'+Acquisition_Processing+'_01_T1_'+band+'.TIF'

for date1 in list_d[:]:
    print('start '+ date1)
    b3path = get_band(date1, 3)
    b5path = get_band(date1, 5)
    with rasterio.open(b3path) as src3:
        b3 = src3.read()
        profile = src3.profile.copy()
    with rasterio.open(b5path) as src5:
        nir = src5.read()
    bands = {'Green': b3.squeeze()/10000, 'Nir': nir.squeeze()/10000}
    wmask = wd.DWImageClustering(bands=bands, bands_keys=['Nir', 'ndwi'], invalid_mask=None, config=config)
    mask = wmask.run_detect_water()
    #plt.imshow(wmask.cluster_matrix)
    #plt.imshow(wmask.water_mask==1)
    #wmask.cluster_matrix.shape
    profile.update({
              'dtype':'float64',
              'height': wmask.cluster_matrix.shape[0],
              'width':wmask.cluster_matrix.shape[1],

           })
    fn = outputpath+b3path[-38:-6]+b3path[-4:]
    with rasterio.open(fn, 'w', **profile) as dst:
        dst.write_band(1, wmask.cluster_matrix)

shp_clip = 'd:\xxx\xxx\salton sea\SaltonSeaArea.shp'
band = b3path
options = gdal.WarpOptions(cutlineDSName=shp_clip,cropToCutline=True)
outBand = gdal.Warp(srcDSOrSrcDSTab= band,
                    destNameOrDestDS=band[:-4]+'_c2'+band[-4:],
                    options=options)



#%%
import cv2
import numpy as np

img = cv2.imread(r'C:\Users\xxx\Desktop\1.png', cv2.IMREAD_COLOR)

h, w, _ = img.shape

GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,binary = cv2.threshold(GrayImage,40,255,cv2.THRESH_BINARY) 

threshold = h/30 * w/30   #设定阈值

#cv2.fingContours寻找图片轮廓信息
"""提取二值化后图片中的轮廓信息 ，返回值contours存储的即是图片中的轮廓信息，是一个向量，内每个元素保存
了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓，有多少轮廓，向量contours就有
多少元素"""
contours,hierarch=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    area = cv2.contourArea(contours[i]) #计算轮廓所占面积
    if area < threshold:
        cv2.drawContours(img,[contours[i]],-1, (84,1,68), thickness=-1)
        continue

cv2.imshow('Output',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r'C:\Users\chenxu\Desktop\1_denoised.png', img) #保存图片
#%%
import landsatxplore
import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import waterdetect as wd
def save_tif(file_tif, file_name, profile_old):
    profile = profile_old.copy()
    profile.update({
          'dtype':'uint8',
          'height': file_tif.shape[0],
          'width':file_tif.shape[1],

    })
    with rasterio.open(file_name, 'w', **profile) as dst:
        dst.write_band(1, file_tif)

username, password = 'Rui7803', '98784usgsUSGS'
api = landsatxplore.api.API(username, password)
Earth_Down = EarthExplorer(username, password)
scenes = api.search(
    dataset='LANDSAT_8_C1',
    latitude=33.17,
    longitude=-115.61,
    start_date='2021-03-20',
    end_date='2021-10-01',
    max_cloud_cover=10)

if len(scenes)>0:
    #Acquisition_Processing = scenes[0]['display_id'].split('039037_')[1][:17]
    Earth_Down.download(identifier=scenes[0]['display_id'], output_dir='E:\SaltonSea039037')


#Acquisition_Processing = '20130324_20170310'
#fpathB3 = 'http://landsat-pds.s3.amazonaws.com/c1/L8/039/037/LC08_L1TP_039037_'+Acquisition_Processing+'_01_T1/LC08_L1TP_039037_'+Acquisition_Processing+'_01_T1_B3.TIF'
#fpathB5 = 'http://landsat-pds.s3.amazonaws.com/c1/L8/039/037/LC08_L1TP_039037_'+Acquisition_Processing+'_01_T1/LC08_L1TP_039037_'+Acquisition_Processing+'_01_T1_B5.TIF'


with rasterio.open(fpathB3) as src_b3:
    b3 = src_b3.read()

with rasterio.open(fpathB5) as src_b5:
    nir = src_b5.read()

bands = {'Green': b3.squeeze()/10000, 'Nir': nir.squeeze()/10000}
wmask = wd.DWImageClustering(bands=bands, bands_keys=['Nir', 'ndwi'], invalid_mask=None, config=config)
mask = wmask.run_detect_water()
plt.imshow(wmask.water_mask==1)
plt.imshow(wmask.cluster_matrix)
plt.title('SaltonSeaMarch2021')
save_tif(acquisition_Processing, src_b3.profile)

config = wd.DWConfig(config_file=r'D:\xxx\xxx\new\WaterDetect.ini')














