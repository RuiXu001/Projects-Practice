# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 19:02:02 2021

@author: xurui


"""
#%%
# pymodis, gcloud, google-cloud-storage, httplib2==0.15, datetime, os, glob, osgeo, gdal, numpy, shapely, geopandas, rioxarray, xarray, subprocess, rasterio
import pymodis
from datetime import timedelta, datetime
import datetime
import re 
from google.cloud import storage
import requests
import glob
import os
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
from osgeo import gdal 
import numpy as np
import subprocess
import rasterio
from rasterio import crs
from rasterio.warp import calculate_default_transform, reproject, Resampling
import shapely 
shapely.speedups.disable()

def c_dir(bdir, filename, format_f):
    f = '.'.join([filename, format_f])
    return os.path.join(bdir, f)

username=''
password=''
token=''
api=''
url_="https://e4ftl01.cr.usgs.gov/MOTA/MCD19A2.006/"
date_obj = datetime.datetime.now()
storage_client = storage.Client.from_service_account_json('d:/msda/data298/new/data298-4fa107054e44.json',project='data298') # 指定凭证文件
bucket = storage_client.get_bucket('data298_2021_rx')
# usgs date in %Y.%m.%d format. eg. 2021.10.26
d_now = date_obj.strftime('%Y.%m.%d/')
output_dir = r'd://tmp'

def download_modis():
    for i in range(7):
        date_tmp = date_obj-timedelta(days=i)
        # check if the file exists in google cloud
        yd = date_tmp.strftime('%Y%j')
        file_pre = 'MODIS/MCD19A2/MCD19A2.A'+str(yd)
        blobs = list(bucket.list_blobs(prefix=file_pre))
        if len(blobs) != 0:
            print(yd, 'file exists in google cloud')
            print(blobs)
            break
        else:
            #file not exists in google cloud, download it 
            d_today = date_obj.strftime('%Y-%m-%d')
            enddate = date_tmp-timedelta(days=1)
            enddate = enddate.strftime('%Y-%m-%d')
            print(d_today, enddate, yd)
            test1 = pymodis.downmodis.downModis(destinationFolder='./', 
                                                password=password,
                                                user=username,
                                                url='https://e4ftl01.cr.usgs.gov',
                                                tiles='h08v05',
                                                path='MOTA',
                                                product='MCD19A2.006',
                                                today=d_today,
                                                enddate=enddate,
                                                jpg=False)# enddate is before today.
            test1.connect()
            test1.downloadsAllDay()
    files = glob.glob('d://tmp//*.hdf')

    for input_f in files:
        print('processing '+input_f.split('.')[1])
        filen1 = input_f.split('\\')[-1][:-4]+'_047'
        filen2 = input_f.split('\\')[-1][:-4]+'_055'
        output_f1 = c_dir(output_dir, filen1, 'tif')
        output_f2 = c_dir(output_dir, filen2, 'tif')
        if os.path.exists(output_f1):
            print(filen1, 'exists')
            pass
        else:
            # tif file with 4 bands
            subprocess.call(f'''gdal_translate HDF4_EOS:EOS_GRID:"'''+input_f+'''":grid1km:Optical_Depth_047 '''+str(output_f1))
            subprocess.call(f'''gdal_translate HDF4_EOS:EOS_GRID:"'''+input_f+'''":grid1km:Optical_Depth_055 '''+str(output_f2))
            
            filen11 = input_f.split('\\')[-1][:-4]+'_047_rep'
            filen22 = input_f.split('\\')[-1][:-4]+'_055_rep'
            output_f11 = c_dir(output_dir, filen11, 'tif')
            output_f22 = c_dir(output_dir, filen22, 'tif')
            
            dst_crs = crs.CRS.from_epsg('32611')#4326是wgs84  # 32611 : UTM 11N for Salton Sea
            
            with rasterio.open(output_f1) as src_ds:
                profile = src_ds.profile
                dst_transform, dst_width, dst_height = calculate_default_transform(src_ds.crs, dst_crs, src_ds.width, src_ds.height, *src_ds.bounds)
                profile.update({'crs': dst_crs,'transform': dst_transform,'width': dst_width,'height': dst_height,'nodata': 0})
                with rasterio.open(output_f11, 'w', **profile) as dst_ds:
                    for i in range(1, src_ds.count + 1):
                        src_array = src_ds.read(i)
                        dst_array = np.empty((dst_height, dst_width), dtype=profile['dtype'])
                        reproject(source=src_array,src_crs=src_ds.crs,src_transform=src_ds.transform,destination=dst_array,dst_transform=dst_transform,dst_crs=dst_crs,resampling=Resampling.cubic,num_threads=2)
                        dst_ds.write(dst_array, i)
                del dst_ds
                        
            with rasterio.open(output_f2) as src_ds:
                profile = src_ds.profile
                dst_transform, dst_width, dst_height = calculate_default_transform(src_ds.crs, dst_crs, src_ds.width, src_ds.height, *src_ds.bounds)
                profile.update({'crs': dst_crs,'transform': dst_transform,'width': dst_width,'height': dst_height,'nodata': 0})
                with rasterio.open(output_f22, 'w', **profile) as dst_ds:
                    for i in range(1, src_ds.count + 1):
                        src_array = src_ds.read(i)
                        dst_array = np.empty((dst_height, dst_width), dtype=profile['dtype'])
                        reproject(source=src_array,src_crs=src_ds.crs,src_transform=src_ds.transform,destination=dst_array,dst_transform=dst_transform,dst_crs=dst_crs,resampling=Resampling.cubic,num_threads=2)
                        dst_ds.write(dst_array, i)
                del dst_ds
                
            filen111 = input_f.split('\\')[-1][:-4]+'_047_rep_f'
            filen222 = input_f.split('\\')[-1][:-4]+'_055_rep_f'
            output_f111 = c_dir(output_dir, filen111, 'tif')
            output_f222 = c_dir(output_dir, filen222, 'tif')
            
            aod = rasterio.open(output_f11)
            profile = aod.profile
            profile['count']=1
            aod_max = np.max(aod.read(), axis=0)
            with rasterio.open(output_f111, 'w', **profile) as outDataRaster:
                outDataRaster.write_band(1, aod_max)
            del outDataRaster
            
            
            aod = rasterio.open(output_f22)
            profile = aod.profile
            profile['count']=1
            aod_max = np.max(aod.read(), axis=0)
            with rasterio.open(output_f222, 'w', **profile) as outDataRaster:
                outDataRaster.write_band(1, aod_max)
            del outDataRaster

    input_shape = r"D:\MSDA\data298\Shapefiles\SaltonSea_l.shp" 
    crop_extent = gpd.read_file(input_shape)
    # input files
    tif_files = glob.glob(output_dir+'//*_rep_f.tif')
    # output
    output_c_tif = output_dir
    for input_f in tif_files:
        # output dir
        output_c_file_n = input_f.split('\\')[-1][:-10]+'_cropped'
        output_c_f_dir = c_dir(output_c_tif, output_c_file_n , 'tif')
        if os.path.exists(output_c_f_dir):
            print(output_c_file_n,' exists')
        else:
            lidar_chm_im = rxr.open_rasterio(input_f, masked=True).squeeze()
            lidar_clipped = lidar_chm_im.rio.clip(crop_extent.geometry.apply(mapping), crop_extent.crs)
            lidar_clipped.rio.to_raster(output_c_f_dir)
            print(output_c_file_n + ' Succeed')

    upload_files = glob.glob('d:\\tmp\\*_cropped.tif')
    for ul_f in upload_files:
        file_name = ul_f.split('tmp\\')[1]
        blob = bucket.blob(f'MODIS/MCD19A2/{file_name}') # 设置 storage 中的文件名
        blob.upload_from_filename(ul_f)#'data298_2021_rx/MODIS/MCD19A2/MCD19A2.A2021299.h08v05.006.2021301061413.hdf')
    file_pre = 'MODIS/MCD19A2/MCD19A2.A'
    blobs = list(bucket.list_blobs(prefix=file_pre))
    print(blobs)

download_modis()
#%%
def download_modis_mod13():
    date_obj
    # check if the file exists in google cloud
    yd = date_obj.strftime('%Y%j')
    file_pre = 'MODIS/MOD13A2/MOD13A2.A'
    blobs = list(bucket.list_blobs(prefix=file_pre))
    last_f = blobs[-1].name.split('h08v05.')[1][4:11]
    last_d = datetime.datetime.strptime(last_f, '%Y%j')
    d_today = date_obj.strftime('%Y-%m-%d')
    enddate = last_d.strftime('%Y-%m-%d')
    print(d_today, enddate, yd)
    test1 = pymodis.downmodis.downModis(destinationFolder='./', 
                                        password=password,
                                        user=username,
                                        url='https://e4ftl01.cr.usgs.gov',
                                        tiles='h08v05',
                                        path='MOLT',
                                        product='MOD13A2.061',
                                        today=d_today,
                                        enddate=enddate,
                                        jpg=False)# enddate is before today.
    test1.connect()
    test1.downloadsAllDay()
    files = glob.glob('d://tmp//*.hdf')

    for input_f in files:
        print('processing '+input_f.split('.')[1])
        filen = input_f.split('\\')[-1][:-4]+'ndvi_rep'
        output_f = c_dir(output_dir, filen, 'tif')
        if os.path.exists(output_f):
            print(filen1, 'exists')
            pass
        else:
            # tif file with 4 bands
            print('Processing ', output_f)
            subprocess.call(f'''gdalwarp HDF4_EOS:EOS_GRID:"'''+input_f+'''":MODIS_Grid_16DAY_1km_VI:"1 km 16 days NDVI" '''+ str(output_f)+ ' -t_srs EPSG:32611')

    input_shape = r"D:\MSDA\data298\Shapefiles\SaltonSea_l.shp" 
    crop_extent = gpd.read_file(input_shape)
    # input files
    tif_files = glob.glob(output_dir+'//*ndvi_rep.tif')
    # output
    output_c_tif = output_dir
    for input_f in tif_files:
        # output dir
        output_c_file_n = input_f.split('\\')[-1][:-10]+'_cropped'
        output_c_f_dir = c_dir(output_c_tif, output_c_file_n , 'tif')
        if os.path.exists(output_c_f_dir):
            print(output_c_file_n,' exists')
        else:
            lidar_chm_im = rxr.open_rasterio(input_f, masked=True).squeeze()
            lidar_clipped = lidar_chm_im.rio.clip(crop_extent.geometry.apply(mapping), crop_extent.crs)
            lidar_clipped.rio.to_raster(output_c_f_dir)
            print(output_c_file_n + ' Succeed')

    upload_files = glob.glob('d:\\tmp\\*rep_cropped.tif')
    for ul_f in upload_files:
        file_name = ul_f.split('tmp\\')[1]
        blob = bucket.blob(f'MODIS/MOD13A2/{file_name}') 
        blob.upload_from_filename(ul_f)
    file_pre = 'MODIS/MOD13A2/MOD13A2.A'
    blobs = list(bucket.list_blobs(prefix=file_pre))
    print(blobs)

download_modis_mod13()


    print(files)
    for input_f in files:
        print('processing '+input_f.split('.')[1])
        filen = input_f.split('\\')[-1][:-4]+'ndvi_rep'
        output_f = c_dir(output_dir, filen, 'tif')
        if os.path.exists(output_f):
            print(filen1, 'exists')
            pass
        else:
            # tif file with 4 bands
            print('Processing ', f'''gdalwarp HDF4_EOS:EOS_GRID:"'''+input_f+'''":MODIS_Grid_16DAY_1km_VI:"1 km 16 days NDVI" '''+ str(output_f)+ ' -t_srs EPSG:32611')
            subprocess.call(f'''gdalwarp HDF4_EOS:EOS_GRID:"'''+input_f+'''":MODIS_Grid_16DAY_1km_VI:"1 km 16 days NDVI" '''+ str(output_f)+ ' -t_srs EPSG:32611')
    print(files)
    files_tif = glob.glob('/tmp/*.hdf')



