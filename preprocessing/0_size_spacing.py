import SimpleITK as sitk
import numpy as np
import os

NII_dir = "/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_9_20210702/rp_im/"


def getNiiList(nii_path):
    nii_list = os.listdir(nii_path)
    return nii_list


def getImageinfo(fName):
    img = sitk.ReadImage(os.path.join(NII_dir, fName))
    origin = img.GetOrigin()
    direction = img.GetDirection()
    spacing = img.GetSpacing()

    img_arr = sitk.GetArrayFromImage(img)
    size = img_arr.shape

    return size, spacing


if __name__ == "__main__":
    i_1 = 0
    i_2 = 0
    i_5 = 0
    j_1 = 0
    j_2 = 0
    all_spacing = []
    nii_list = getNiiList(NII_dir)
    # nii_list = ['old-6029230-6624449.nii.gz']
    total = len(nii_list)
    print("total: ", total)
    for i in range(total):
        _niiName = nii_list[i]
        size, spacing = getImageinfo(_niiName)
        all_spacing.append(spacing[0])
        # spacing_z = spacing[2]
        # if spacing_z == 0.625:
        #     i_1 = i_1 + 1
        # elif spacing_z == 2.0:
        #     i_2 = i_2 + 1
        # elif spacing_z == 5.0:
        #     i_5 = i_5 + 1
        # else:
        #     print(spacing_z)

        # j = i
        # size_xy = size[2]
        # if size_xy < 448:
        #     j_1 = j_1 + 1
        #     print("==============================", _niiName)
        # elif size_xy >= 448:
        #     j_2 = j_2 + 1
        # else:
        #     print(size_xy)

        # with open('/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/zheda_normal_imgs_resample_all_result.txt', 'a') as file_handle:
        #     file_handle.write('File {} info [{} , {}]\n'.format(_niiName, size, spacing))  # 此时不需在第2行中的转为字符串
        print('File {} info [{} , {}]'.format(_niiName, size, spacing))
    print('spacing x, y : mean {} min-max [{}-{}]'.format(np.mean(all_spacing), np.min(all_spacing), np.max(all_spacing)))
    # print('spacing z : [0.625[{}],2.0[{}],5.0[{}]]'.format(i_1, i_2, i_5))
    # print('size_xy : [<448[{}],>=448[{}]]'.format(j_1, j_2))