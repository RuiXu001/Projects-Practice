# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:25:42 2022

@author: xurui
"""

import os
import cv2
from colorama import Fore, Back, Style
import numpy as np
import sys

basedir = r'D:/research/drone/UAV-benchmark-M/M0101'
os.chdir(basedir)

class VideoCombiner(object):
    def __init__(self, img_dir, f_name):
        self.img_dir = img_dir
        self.f_name = f_name
        if not os.path.exists(self.img_dir):
            print(Fore.RED + '=> Error: ' + '{} not exist.'.format(self.img_dir))
            exit(0)
        self._get_video_shape()
    def _get_video_shape(self):
        self.all_images = [os.path.join(self.img_dir, i) for i in os.listdir(self.img_dir)]
        sample_img = np.random.choice(self.all_images)
        if os.path.exists(sample_img):
            img = cv2.imread(sample_img)
            self.video_shape = img.shape
        else:
            print(Fore.RED + '=> Error: ' + '{} not found or open failed, try again.'.format(sample_img))
            exit(0)
    def combine(self):
        target_file='{}.mp4'.format(self.f_name)
        size = (self.video_shape[1], self.video_shape[0])
        print('=> target video frame size: ', size)
        print('=> all {} frames to solve.'.format(len(self.all_images)))
        video_writer = cv2.VideoWriter(target_file, cv2.VideoWriter_fourcc(*'DIVX'), 24, size) # 24 fps/s
        i = 0
        print('=> Solving, be patient.')
        for img in self.all_images:
            img = cv2.imread(img, cv2.COLOR_BGR2RGB)
            i += 1
            # print('=> Solving: ', i)
            for i in range(2):
                video_writer.write(img)
        video_writer.release()
        print('Done!')

# d = sys.argv[1]
combiner = VideoCombiner(r'D:/research/drone/UAV-benchmark-M/M0101/', 'M0101')
combiner.combine()