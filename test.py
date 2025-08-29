import matplotlib
matplotlib.use('TkAgg')

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

WeaponLists = glob.glob("WeaponList/*")

img = cv2.imread('testimage.png')
for weapon in WeaponLists:    
    template = cv2.imread(weapon) 
    result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    th, tw = template.shape[:2]
    threshold = 0.99
    loc = np.where(result >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th), (255,0,255), 2)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("result", img2)
cv2.waitKey(0)