import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt

WeaponLists = glob.glob("WeaponList/*")
resultlist = np.array([])

img_rgb = cv.imread('source/sample2.png',cv.IMREAD_COLOR_BGR)
assert img_rgb is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

for weapon_dir in WeaponLists:    
    template = cv.imread(weapon_dir, cv.IMREAD_GRAYSCALE) 
    assert template is not None, "file could not be read, check with os.path.exists()"

    #may need to change
    result = cv.matchTemplate(img_gray, template, cv.TM_CCORR_NORMED)
    th, tw = template.shape[:2]
    threshold = 0.85
    loc = np.where(result >= threshold)
    w, h = template.shape

    # remove covering selection
    prev_loc = 0
    loc_list_x = loc[0]
    loc_list_y = loc[1]
    #print(loc_list_x, loc_list_y)   #for test

    if len(loc[0])>1:
        counter = 0
        loc_copy = loc_list_x
        for loc_item in loc_copy:
            #print(counter, prev_loc, loc_item)
            if abs(loc_item-prev_loc) <= 10:
                # delete selected.
                loc_list_x = np.delete(loc_list_x, counter)
                loc_list_y = np.delete(loc_list_y, counter)
                #print("delete")
            else:
                counter+=1
                prev_loc = loc_item
                #print("add counter : ", counter)

            if prev_loc == 0:
                prev_loc = loc_item            

    # update array for results
    weapon_filename = weapon_dir.replace("WeaponList\\", "")
    weapon_name = weapon_filename.replace(".png", "")
    #print(weapon_name,len(loc_list_x))
    resultlist = np.append(resultlist, [weapon_name, len(loc_list_x)])

    # make image for test
    location_list = (loc_list_x, loc_list_y)
    print(weapon_name, location_list)
    for pt in zip(*location_list[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
 
cv.imwrite('result/mul_res.png',img_rgb)

resultlist = resultlist.reshape(-1, 2)
print(resultlist)
# todo : separate weaponname and num by colon
np.savetxt("result/result.csv",resultlist, fmt='%s')