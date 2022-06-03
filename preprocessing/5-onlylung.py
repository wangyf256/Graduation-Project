# -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image

mask_image_path = "/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/radiopaedia_Lung_Mask_use_flip_resample_crop_png_812/"  # 存放所有标注图片的文件夹
image_path = "/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/COVID-19-CT-Seg_radiopaedia_flip_resample_crop_png_812/"  # 存放所有原图的文件夹
onlylung_path = "/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/testimg-onlylung_812/"  # 存放所有在原图叠加标注轮廓图片的文件夹

if not os.path.isdir(onlylung_path):
    os.makedirs(onlylung_path)


def getNameList(path):
    fNameList = os.listdir(path)
    # print(fNameList)
    return fNameList

if __name__ == "__main__":
    mask_image_list = getNameList(mask_image_path)
    # print(ann_image_list)
    image_list = getNameList(image_path)
    # print(image_list)

    for f_mask in mask_image_list:
        f = f_mask
        f_onlylung = f_mask
        image = image_path + f
        mask_image = mask_image_path + f_mask
        onlylung_image = onlylung_path + f_onlylung
        img2D = np.array(
            Image.open(image).convert('L'))  # dtype "uint8"

        mask2D = np.array(
            Image.open(mask_image).convert('L'))  # dtype "uint8"

        mask2D[mask2D > 0] = 1
        onlylung2D = img2D * mask2D

        onlylung2D = Image.fromarray(onlylung2D.astype(np.uint8))  # mode "L"
        onlylung2D.save(onlylung_image)
        print("saved", f_onlylung)




