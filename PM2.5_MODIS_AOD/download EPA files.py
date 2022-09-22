# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:08:56 2021

@author: Rui7803
"""

import urllib.request
import os
import re
D_dir = input('Please input your work path:')
# D_dir = 'c:\xxx\xxx\salton sea'
os.chdir(D_dir)

# hourly data from 1980 - 2020
o3 = 44201
so2 = 42401
co = 42101
no = 42602
PM25_FRM_FEM = 88101
PM25_nonFRM_FEM = 88502
PM10 = 81102	
PM25S = 'SPEC'	
PM10S = 'PM10SPEC'

Wind = 'WIND'
T = 'TEMP'
Pressure = 'PRESS' # Barometric Pressure (64101)	
RH_DP = 'RH_DP' # RH and Dewpoint

identifer_list = [o3, so2, co, no, 
                  PM25_FRM_FEM, PM25_nonFRM_FEM, PM10, PM25S, PM10S, 
                  Wind, T, Pressure, RH_DP]
url_h = 'https://aqs.epa.gov/aqsweb/airdata/hourly_GAS_YYYY.zip'


def download_fs(url):
    file_name = url.split('/')[-1]
    if os.path.exists(file_name):
        print(file_name + ' exists. ') # check if the file exists. 
    else:
        print('Downloading '+file_name)
        try:
            urllib.request.urlretrieve(url, file_name)
        except:
            print('Error happened when downloading '+file_name)
            pass
        



# target_pollutant = o3 # get the number of target pollutant

for i in identifer_list:
    for j in range(1980, 2021):
        url = url_h.replace('GAS', str(i)).replace('YYYY', str(j))
        # print(url)
        download_fs(url)
print('Done!')




