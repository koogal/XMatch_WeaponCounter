#import matplotlib
#matplotlib.use('TkAgg')
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

WeaponLists = glob.glob("WeaponList/*")     # 武器テンプレート一覧

SourceImages = glob.glob("source/*.png")    # source配下のすべての画像
for source_path in SourceImages:   
    img = cv2.imread(source_path)
    assert img is not None, f"{source_path} が読み込めません"

    for weapon in WeaponLists:
        template = cv2.imread(weapon)
        assert template is not None, f"{weapon} が読み込めません"

        result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        th, tw = template.shape[:2]
        threshold = 0.98
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th), (255, 0, 255), 2)

# 結果画像の保存
    filename = os.path.basename(source_path)
    output_path = os.path.join("result", f"{filename}_result.png")
    cv2.imwrite(output_path, img)
