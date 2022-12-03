# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:26:55 2022

@author: xurui
"""

from PIL import Image

# inpath = 'C:/Users/xurui/OneDrive/桌面/0000001.jpg'
# outpath = 'C:/Users/xurui/OneDrive/桌面/0000001_l.jpg'

def lower(inpath, outpath, scale):
    img = Image.open(inpath)
    img.show()
    pix = img.load()
    
    width, height = img.size
    
    type = img.format
    out = img.resize((int(width/scale), int(height/scale)), Image.ANTIALIAS)
    out.show()
    
    out.save(outpath, type)
