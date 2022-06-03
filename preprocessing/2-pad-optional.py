import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm

mask_pad_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/3_test_nodule/lung-mask-nii_resample_pad/'
if not os.path.isdir(mask_pad_path):
    os.makedirs(mask_pad_path)
imgs_pad_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/3_test_nodule/nii_resample_pad/'
if not os.path.isdir(imgs_pad_path):
    os.makedirs(imgs_pad_path)

masks_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/3_test_nodule/lung-mask-nii_resample/'
imgs_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/3_test_nodule/nii_resample/'


def getNiiList(nii_path):
    nii_list = os.listdir(nii_path)
    return nii_list


def pad(crop_3D):
    mask_0 = np.zeros((crop_3D.shape[0], 448, 448))
    # mask_0 = np.full((crop_3D.shape[0], crop_3D.shape[1], 448), -1350)

    print("origin dtype: ", crop_3D.dtype, mask_0.dtype)  # float64 float64
    print("origin shape:", crop_3D.shape, mask_0.shape)

    mask_minz = min(crop_3D.shape[0], mask_0.shape[0])
    mask_miny = min(crop_3D.shape[1], mask_0.shape[1])
    mask_minx = min(crop_3D.shape[2], mask_0.shape[2])

    # offsetz, offsety, offsetx = 0, 0, 0
    # len_z, len_y, len_x = crop_3D.shape[0], crop_3D.shape[1], crop_3D.shape[2]
    # if crop_3D.shape[0] > mask_0.shape[0]:
    #     offsetz = (crop_3D.shape[0] - mask_0.shape[0]) // 2
    #     len_z = mask_0.shape[0]
    # if crop_3D.shape[1] > mask_0.shape[1]:
    #     offsety = (crop_3D.shape[1] - mask_0.shape[1]) // 2
    #     len_y = mask_0.shape[1]
    # if crop_3D.shape[2] > mask_0.shape[2]:
    #     offsetx = (crop_3D.shape[2] - mask_0.shape[2]) // 2
    #     len_x = mask_0.shape[2]
    # mask_used = crop_3D[offsetz:len_z + offsetz, offsety:len_y + offsety, offsetx:len_x + offsetx]

    mask_used = crop_3D

    # print(crop_3D.shape, mask_0.shape, mask_used.shape, "shape------------------")
    for i in range(0, mask_minz):
        offset_z = (mask_0.shape[0] - mask_minz) // 2
        for j in range(0, mask_miny):
            offset_y = (mask_0.shape[1] - mask_miny) // 2
            for k in range(0, mask_minx):
                offset_x = (mask_0.shape[2] - mask_minx) // 2
                mask_0[offset_z + i][offset_y + j][offset_x + k] = mask_used[i][j][k]
    print("pad dtype: ", crop_3D.dtype, mask_0.dtype)  # float64 float64
    print("pad shape:", crop_3D.shape, mask_0.shape)
    return mask_0


def save_volume(crop_3D_arr, imgs_pad_path, _volume_info, name):
    crop_3D = sitk.GetImageFromArray(crop_3D_arr)
    crop_3D.SetOrigin(_volume_info[0])
    crop_3D.SetDirection(_volume_info[1])
    crop_3D.SetSpacing(_volume_info[2])
    sitk.WriteImage(crop_3D, os.path.join(imgs_pad_path, name))


def all_data(nii_list):
    for i in tqdm(range(len(nii_list))):
        # mask和img名字一样
        masks_imgs_name = nii_list[i]
        print("\n")
        print(masks_imgs_name)

        # 读取 (Width, Height, Depth)
        mask = sitk.ReadImage(os.path.join(masks_path, masks_imgs_name))
        img = sitk.ReadImage(os.path.join(imgs_path, masks_imgs_name))

        # imgs_img的一些参数
        origin = img.GetOrigin()
        direction = img.GetDirection()
        spacing = img.GetSpacing()
        _volume_info = [origin, direction, spacing]

        # 转为Array (Depth, Width, Height)
        mask_arr = sitk.GetArrayFromImage(mask)
        print("mask max, min:", np.max(mask_arr), np.min(mask_arr))
        # pad 为统一大小
        mask_pad_volume = pad(mask_arr)
        print("pad mask max, min:", np.max(mask_pad_volume), np.min(mask_pad_volume))
        save_volume(mask_pad_volume, mask_pad_path, _volume_info, masks_imgs_name)

        img_arr = sitk.GetArrayFromImage(img)
        print("img max, min:", np.max(img_arr), np.min(img_arr))
        img_pad_volume = pad(img_arr)
        print("pad img max, min:", np.max(img_pad_volume), np.min(img_pad_volume))
        save_volume(img_pad_volume, imgs_pad_path, _volume_info, masks_imgs_name)


if __name__ == "__main__":
    # nii_list = getNiiList(masks_path)
    # nii_list = ['Mul_GGO_018.nii.gz', 'Mul_GGO_019.nii.gz']
    nii_list = ['4381507.nii.gz', '4382521.nii.gz', '2390276.nii.gz', '4082259.nii.gz', '1264874.nii.gz', '3589885.nii.gz', '2598381.nii.gz']
    all_data(nii_list)
