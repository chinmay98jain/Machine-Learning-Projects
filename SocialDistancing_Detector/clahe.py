# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 02:08:00 2021

@author: chinm
"""

import numpy as np
import cv2 as cv
img=cv.imread('stadium.jpg',0)
cv.waitKey(0)
clahe=cv.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
cl11=clahe.apply(img)
cv.imwrite('stadium_clahe.jpeg',cl11)
cv.imshow('CLAHE Enhanced Image',cl11)
cv.waitKey(0)
equ=cv.equalizeHist(img)
res=np.hstack((img,equ,cl11))
cv.imshow('Enhanced Images',res)
cv.waitKey(0)