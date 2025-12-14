import os
import cv2 as cv
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
import summary

# -----------------------------
# テンプレート事前ロード（グローバル）
# -----------------------------
WeaponLists = glob.glob("WeaponList/*")
templates = []

for weapon_dir in WeaponLists:
    template = cv.imread(weapon_dir, cv.IMREAD_GRAYSCALE)
    assert template is not None, f"{weapon_dir} が読み込めません"
    weapon_name = os.path.basename(weapon_dir).replace(".png", "")
    templates.append((weapon_name, template))


# -----------------------------
# 並列で実行する関数
# -----------------------------
def process_source(source_path):
    img_rgb = cv.imread(source_path, cv.IMREAD_COLOR)
    assert img_rgb is not None, f"{source_path} が読み込めません"
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    source_name = os.path.basename(source_path).replace(".png", "")
    resultlist = []

    for weapon_name, template in templates:

        result = cv.matchTemplate(img_gray, template, cv.TM_CCORR_NORMED)
        th, tw = template.shape[:2]
        threshold = 0.98
        loc = np.where(result >= threshold)
        w, h = template.shape

        # 重複除去
        prev_loc = 0
        loc_list_x = loc[0]
        loc_list_y = loc[1]

        if len(loc_list_x) > 1:
            counter = 0
            loc_copy = loc_list_x.copy()
            for loc_item in loc_copy:
                if abs(loc_item - prev_loc) <= 10:
                    loc_list_x = np.delete(loc_list_x, counter)
                    loc_list_y = np.delete(loc_list_y, counter)
                else:
                    counter += 1
                    prev_loc = loc_item

                if prev_loc == 0:
                    prev_loc = loc_item

        resultlist.append((weapon_name, len(loc_list_x)))

        # rectangle 描画
        for pt in zip(loc_list_y, loc_list_x):
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    # 画像保存
    output_name = f"{source_name}_res.png"
    cv.imwrite(f"result/{output_name}", img_rgb)

    # CSV 保存
    formatted_result = [f"{name}:{count}" for name, count in resultlist]
    output_csv = f"{source_name}_result.csv"
    np.savetxt(f"result/{output_csv}", formatted_result, fmt='%s')

    return source_name


# -----------------------------
# 並列実行
# -----------------------------
if __name__ == "__main__":
    SourceImages = glob.glob("source/*.png")

    with ProcessPoolExecutor() as executor:
        list(executor.map(process_source, SourceImages))

    summary.generate_summary()