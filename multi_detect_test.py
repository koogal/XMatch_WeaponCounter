import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
img_rgb = cv.imread('testimage.png', cv.IMREAD_ANYDEPTH)
assert img_rgb is not None, "file could not be read, check with os.path.exists()"
template = cv.imread('WeaponList/Mint_Decavitator.png', cv.IMREAD_ANYDEPTH)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]
 
res = cv.matchTemplate(img_rgb,template,cv.TM_CCOEFF_NORMED)
threshold = 0.99
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
 
cv.imwrite('result/mul_res.png',img_rgb)