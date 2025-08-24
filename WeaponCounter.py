import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt

WeaponLists = glob.glob("WeaponList/*")
resultlist = np.array([])

img_rgb = cv.imread('source/testimage.png',cv.IMREAD_COLOR_BGR)
assert img_rgb is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

for weapon_dir in WeaponLists:    
    template = cv.imread(weapon_dir, cv.IMREAD_GRAYSCALE) 
    assert template is not None, "file could not be read, check with os.path.exists()"

    result = cv.matchTemplate(img_gray, template, cv.TM_CCORR_NORMED)
    th, tw = template.shape[:2]
    threshold = 0.95
    loc = np.where(result >= threshold)
    w, h = template.shape

    weapon_filename = weapon_dir.replace("WeaponList\\", "")
    weapon_name = weapon_filename.replace(".png", "")
    print(weapon_name,len(loc[0]))
    resultlist = np.append(resultlist, [weapon_name, len(loc[0])])

resultlist = resultlist.reshape(-1, 2)
print(resultlist)
np.savetxt("result/result.csv",resultlist, fmt='%s')