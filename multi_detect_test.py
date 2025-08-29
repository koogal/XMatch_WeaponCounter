import cv2 as cv
import numpy as np
import glob
#from matplotlib import pyplot as plt
import os

 
image_paths = glob.glob("source/*.png")
assert image_paths is not None, "file could not be read, check with os.path.exists()"

template_color = cv.imread('WeaponList/Forge_Splattershot_Pro.png', cv.IMREAD_COLOR)
assert template_color is not None, "file could not be read, check with os.path.exists()"

template = cv.cvtColor(template_color, cv.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

threshold = 0.99
 
for img_path in image_paths:
    img_rgb = cv.imread(img_path, cv.IMREAD_COLOR)
    assert img_rgb is not None, f"{img_path} が読み込めませんでした"
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)


    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
 
    filename = os.path.basename(img_path)
    save_path = f"result/result_{filename}"
    cv.imwrite(save_path, img_rgb)
