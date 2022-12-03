# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 19:30:53 2022

@author: xurui
"""

import zipfile
import os
import glob
from pathlib import Path

base = 'D:/research/drone/crack'
compressed_lst = ['gz', 'tar', 'zip', 'rar']
folders = glob.glob(base+'/*')

for folder in folders:
    files = glob.glob(folder + '/*')
    files = [i for i in files if (i.split('.')[-1] in compressed_lst)]
    for f in files:
        zFile = zipfile.ZipFile(f)
        print('Unzip ', f)
        for fileM in zFile.namelist(): 
            zFile.extract(fileM, str(Path(f).resolve().parent))
        zFile.close()


f = 'D:/research/drone/crack/RDD2020'









