import SimpleITK as sitk
import os
from tqdm import tqdm
import numpy as np

masks_resample_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/radiopaedia_Lung_Mask_use_flip_resample/'
if not os.path.isdir(masks_resample_path):
    os.makedirs(masks_resample_path)
imgs_resample_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/COVID-19-CT-Seg_radiopaedia_flip_resample/'
if not os.path.isdir(imgs_resample_path):
    os.makedirs(imgs_resample_path)

# masks_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/zhedaNormalMedian210128/zheda_mask_bbox/'
# imgs_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/zhedaNormalMedian210128/zheda_imgs_bbox/'


masks_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/radiopaedia_Lung_Mask_use_flip/'
imgs_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/COVID-19-CT-Seg_radiopaedia_flip/'

def getNiiList(nii_path):
    nii_list = os.listdir(nii_path)
    return nii_list


def Resampling(img, lable=False):
    # 设置图像新的分辨率
    new_spacing = (0.6738280057907104, 0.6738280057907104, img.GetSpacing()[2])
    # 计算图像在新的分辨率下尺寸大小
    new_size = [int(round(img.GetSize()[0] * (img.GetSpacing()[0] / 0.6738280057907104))),
                int(round(img.GetSize()[1] * (img.GetSpacing()[1] / 0.6738280057907104))),
                int(round(img.GetSize()[2] * (img.GetSpacing()[2] / img.GetSpacing()[2])))]

    resampleSliceFilter = sitk.ResampleImageFilter()  # 初始化

    if lable == False:
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkBSpline,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
        # ResampleimageArray = sitk.GetArrayFromImage(Resampleimage)
        # ResampleimageArray[ResampleimageArray < 0] = 0  # 将图中小于0的元素置为0
    else:  # for label, should use sitk.sitkLinear to make sure the original and resampled label are the same!!!
        Resampleimage = resampleSliceFilter.Execute(img, new_size, sitk.Transform(), sitk.sitkLinear,
                                                    img.GetOrigin(), new_spacing, img.GetDirection(), 0,
                                                    img.GetPixelIDValue())
        # ResampleimageArray = sitk.GetArrayFromImage(Resampleimage)

    return Resampleimage


def all_data(nii_list):
    for i in tqdm(range(len(nii_list))):
        masks_imgs_name = nii_list[i]
        print("\n")
        print(masks_imgs_name)

        mask = sitk.ReadImage(os.path.join(masks_path, masks_imgs_name))
        img = sitk.ReadImage(os.path.join(imgs_path, masks_imgs_name))
        print("msak size:", mask.GetSize())
        print("img size:", img.GetSize())  # (Width, Height, Depth)

        mask_arr = sitk.GetArrayFromImage(mask)
        img_arr = sitk.GetArrayFromImage(img)
        print("mask max, min:", np.max(mask_arr), np.min(mask_arr))
        print("img max, min:", np.max(img_arr), np.min(img_arr))

        resample_mask = Resampling(mask, lable=True)
        resample_img = Resampling(img, lable=False)
        print("resample msak size:", resample_mask.GetSize())
        print("resample img size:", resample_img.GetSize())

        resample_mask_arr = sitk.GetArrayFromImage(resample_mask)
        resample_img_arr = sitk.GetArrayFromImage(resample_img)
        print("resample mask max, min:", np.max(resample_mask_arr), np.min(resample_mask_arr))
        print("resample img max, min:", np.max(resample_img_arr), np.min(resample_img_arr))

        _volume_info_img = [img.GetOrigin(), img.GetDirection(), img.GetSpacing()]
        _volume_info_resample_img = [resample_img.GetOrigin(), resample_img.GetDirection(), resample_img.GetSpacing()]
        print("volume_info1-1:", _volume_info_img)
        print("volume_info2-1:", _volume_info_resample_img)

        sitk.WriteImage(resample_mask, os.path.join(masks_resample_path, masks_imgs_name))
        # sitk.WriteImage(resample_img, os.path.join(imgs_resample_path, masks_imgs_name))


if __name__ == "__main__":
    nii_list = getNiiList(masks_path)
    # nii_list = ['Mul_CON_026.nii.gz', 'Ret_GGO_018.nii.gz', 'Mul_GGO_018.nii.gz', 'Mul_GGO_019.nii.gz', 'Mul_CON_007.nii.gz', 'Ret_GGO_009.nii.gz', 'HCM_027.nii.gz']
    all_data(nii_list)
