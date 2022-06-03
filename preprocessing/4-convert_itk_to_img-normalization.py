import SimpleITK as sitk
import numpy as np
import os
import sys
from tqdm import tqdm
from PIL import Image
import cv2

# img_dir = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/2_test_zheda_ann_COVID19/ann_COVID19_wholelung/testimg-onlylung/test/2'
#
mask_img_dir = 'D:/data-20220323/origin-seg-png/'
if not os.path.isdir(mask_img_dir):
    os.makedirs(mask_img_dir)
# img_img_dir = 'D:/data-20220323/origin_png/'
# if not os.path.isdir(img_img_dir):
#     os.makedirs(img_img_dir)

mask_nii_dir = 'D:/data-20220323/origin-seg/'
# img_nii_dir = 'D:/data-20220323/origin/'

# if not os.path.isdir(img_dir):
#     os.makedirs(img_dir)


def getNiiList(nii_path):
    nii_list = os.listdir(nii_path)
    return nii_list


def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


def save_img_only_foreground(mask3D_arr, img3D_arr, name):
    WIN_LEVEL = -600
    WIN_SIZE = 1500
    minCTValue = WIN_LEVEL - WIN_SIZE / 2  # -1350
    maxCTValue = minCTValue + WIN_SIZE  # 150

    print("3D mask max, min:", np.max(mask3D_arr), np.min(mask3D_arr))
    print("3D img max, min:", np.max(img3D_arr), np.min(img3D_arr))

    for z in range(mask3D_arr.shape[0]):
        mask_2D = mask3D_arr[z, :, :]
        mask_2D[mask_2D > 0] = 1
        img_2D = img3D_arr[z, :, :]

        print("51--2D img max, min:", np.max(img_2D), np.min(img_2D))
        img_2D = img_2D * mask_2D
        print("53--2D img max, min:", np.max(img_2D), np.min(img_2D))
        mask_2D[mask_2D == 0] = -1350  # -1350/-1024
        mask_2D[mask_2D == 1] = 0
        img_2D += mask_2D
        # print(img_2D)
        print("57--2D img max, min:", np.max(img_2D), np.min(img_2D))
        img_2D[img_2D <= minCTValue] = minCTValue
        img_2D[img_2D >= maxCTValue] = maxCTValue
        img_2D = (img_2D - minCTValue) / WIN_SIZE * 255.0

        # print(img_2D)

        dump_path = os.path.join(img_dir, name + "-" + str(z) + ".png")

        # 方法1
        # cv.imwrite(dump_path, img_2D)
        # img_2D_read = cv.imread(dump_path, cv.IMREAD_GRAYSCALE)

        # 方法2
        print("71--2D img max, min:", np.max(img_2D), np.min(img_2D))
        img_2D = Image.fromarray(img_2D.astype(np.uint8))  # mode "L"
        img_2D.save(dump_path)
        img_2D_read = np.array(
            Image.open(dump_path).convert('L'))  # dtype "uint8"
        print("74--2D img max, min:", np.max(img_2D_read), np.min(img_2D_read))


def all_data(nii_list):
    for i in tqdm(range(len(nii_list))):
        # mask和img名字一样
        masks_imgs_name = nii_list[i]
        print("\n")
        print(masks_imgs_name)

        # 读取 (Width, Height, Depth)
        mask3D = sitk.ReadImage(os.path.join(mask_nii_dir, masks_imgs_name))
        img3D = sitk.ReadImage(os.path.join(img_nii_dir, masks_imgs_name))

        # 转为Array (Depth, Width, Height)
        mask3D_arr = sitk.GetArrayFromImage(mask3D)
        img3D_arr = sitk.GetArrayFromImage(img3D)

        save_img_only_foreground(
            mask3D_arr,
            img3D_arr,
            masks_imgs_name.strip(".nii.gz"))


def save_mask(mask3D_arr, name):
    for z in range(mask3D_arr.shape[0]):
        mask_2D = mask3D_arr[z, :, :]
        print("85--2D mask max, min:", np.max(mask_2D), np.min(mask_2D))
        mask_2D[mask_2D > 0] = 255

        dump_path = os.path.join(mask_img_dir, name + "-" + str(z+1) + ".png")

        # cv.imwrite(dump_path, mask_2D.astype(np.uint8))
        # mask_2D_read = cv.imread(dump_path, cv.IMREAD_GRAYSCALE)

        # img_2D = np.rot90(img_2D, k=2)
        # mask_2D = np.flip(mask_2D, axis=0)
        # mask_2D = cv2.resize(mask_2D, (448, 320))

        mask_2D = Image.fromarray(mask_2D.astype(np.uint8))  # mode "L"
        mask_2D.save(dump_path)
        mask_2D_read = np.array(
            Image.open(dump_path).convert('L'))  # dtype "uint8"
        print(
            "93--2D mask max, min:",
            np.max(mask_2D_read),
            np.min(mask_2D_read))


def all_data_mask(nii_list):
    for i in tqdm(range(len(nii_list))):
        # mask和img名字一样
        masks_imgs_name = nii_list[i]
        print(masks_imgs_name)

        # 读取 (Width, Height, Depth)
        mask3D = sitk.ReadImage(os.path.join(mask_nii_dir, masks_imgs_name))

        # 转为Array (Depth, Width, Height)
        mask3D_arr = sitk.GetArrayFromImage(mask3D)

        save_mask(mask3D_arr, masks_imgs_name.strip(".nii.gz"))


def save_img(img3D_arr, name):
    WIN_LEVEL = -600
    WIN_SIZE = 1500
    minCTValue = WIN_LEVEL - WIN_SIZE / 2  # -1350
    maxCTValue = minCTValue + WIN_SIZE  # 150

    # level = (min+max)/2  window = max-min

    for z in range(img3D_arr.shape[0]):
        img_2D = img3D_arr[z, :, :]

        print("145--2D ori img max, min:", np.max(img_2D), np.min(img_2D))
        img_2D[img_2D <= minCTValue] = minCTValue
        img_2D[img_2D >= maxCTValue] = maxCTValue
        img_2D = (img_2D - minCTValue) / WIN_SIZE * 255.0
        print(
            "149--2D normlization img max, min:",
            np.max(img_2D),
            np.min(img_2D))

        dump_path = os.path.join(img_img_dir, name + "-" + str(z+1) + ".png")

        # cv.imwrite(dump_path, img_2D.astype(np.uint8))
        # img_2D_read = cv.imread(dump_path, cv.IMREAD_GRAYSCALE)

        # img_2D = np.rot90(img_2D, k=2)
        # img_2D = np.flip(img_2D, axis=0)

        # print("before resize max, min:", np.max(img_2D), np.min(img_2D))
        # img_2D = cv2.resize(img_2D, (448, 320))
        # print("resize img max, min:", np.max(img_2D), np.min(img_2D))

        img_2D = Image.fromarray(img_2D.astype(np.uint8))  # mode "L"
        img_2D.save(dump_path)
        img_2D_read = np.array(
            Image.open(dump_path).convert('L'))  # dtype "uint8"
        print(
            "159--2D read img max, min:",
            np.max(img_2D_read),
            np.min(img_2D_read))


def all_data_img(nii_list):
    for i in tqdm(range(len(nii_list))):
        # mask和img名字一样
        masks_imgs_name = nii_list[i]
        print(masks_imgs_name)

        # 读取 (Width, Height, Depth)
        img3D = sitk.ReadImage(os.path.join(img_nii_dir, masks_imgs_name))

        # 转为Array (Depth, Width, Height)
        img3D_arr = sitk.GetArrayFromImage(img3D)

        save_img(img3D_arr, masks_imgs_name.strip(".nii.gz"))


if __name__ == "__main__":
    # nii_list = getNiiList(mask_nii_dir)
    # # nii_list = ['middle-380021-6624435.nii.gz']
    # # print(nii_list)
    # print("only lung")
    # all_data(nii_list)


    nii_list = getNiiList(mask_nii_dir)
    print("only mask")
    all_data_mask(nii_list)

    # nii_list = getNiiList(img_nii_dir)
    # print("only img")
    # all_data_img(nii_list)
