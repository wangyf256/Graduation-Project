import SimpleITK as sitk
import os
from tqdm import tqdm
import numpy as np

'''
预处理步骤：
1.根据肺部mask取最小bounding box并裁剪mask和img
2.对需要重采样的数据进行重采样（可选）
3.由于重采样之后数据大小不一，应该把所有数据pad成一样大，这里要pad成的大小需要能够包含所有重采样后的数据大小
4.过CT窗，并存成图片，应关注归一化后的最大最小值
所有的结果存在  >/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/zhedaNormal210128/x.txt

在用整图训练时，有pad的白边可能影响训练，因此改变预处理步骤
2.重采样
1.pad （一般用不到，只有个别resample完不足448*320的需要）
3.bbox
'''

masks_boundingbox_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/radiopaedia_Lung_Mask_use_flip_resample_crop/'
imgs_boundingbox_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/radiopaedia_Infection_Mask_flip_resample_crop/'

if not os.path.isdir(masks_boundingbox_path):
    os.makedirs(masks_boundingbox_path)
if not os.path.isdir(imgs_boundingbox_path):
    os.makedirs(imgs_boundingbox_path)

masks_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/radiopaedia_Lung_Mask_use_flip_resample/'
imgs_path = '/data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/test_covid19_radiopaedia/radiopaedia_Infection_Mask_flip_resample/'


def getNiiList(nii_path):
    nii_list = os.listdir(nii_path)
    return nii_list


def cul_bounding_box(masks):
    # 自己写的，三层循环计算较慢
    cro_z = []
    cro_y = []
    cro_x = []
    for z in range(masks.shape[0]):
        for y in range(masks.shape[1]):
            for x in range(masks.shape[2]):
                if masks[z][y][x] > 0:
                    cro_z.append(z)
                    cro_y.append(y)
                    cro_x.append(x)

    zz = np.array(cro_z).max() - np.array(cro_z).min()
    yy = np.array(cro_y).max() - np.array(cro_y).min()
    xx = np.array(cro_x).max() - np.array(cro_x).min()

    centery = (np.array(cro_y).max() + np.array(cro_y).min())//2
    centerx = (np.array(cro_x).max() + np.array(cro_x).min())//2

    # Depth, Height, Width, centery, centerx 239 306 469 269 307

    return zz, yy, xx, centery, centerx


def find_bounding_box(mask):
    # 利用simpleitk,计算较快
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    labelFilter = sitk.LabelShapeStatisticsImageFilter()
    labelFilter.Execute(mask)
    bbox = labelFilter.GetBoundingBox(1)
    # 参数1代表标签值，意思是获取标签值为1的区域所在的bounding box
    # bbox构成(起点坐标[X,Y,Z],边长[w,h,d])
    # centerz = bbox[2] + bbox[5] // 2
    # centery = bbox[1] + bbox[4] // 2
    # centerx = bbox[0] + bbox[3] // 2
    # print("centerz, centery, centerx: ", centerz, centery, centerx)
    # bbox(x, y, z, w, h, d): (73, 116, 0, 470, 307, 240)
    # centery, centerx: 269 308

    return bbox


def crop_center(img, centerx, centery, centerz, width, height, depth):
    # return img[:, centery-cropy//2:centery + cropy//2, centerx-cropx//2:centerx + cropx//2]
    if img.shape[1] < height or img.shape[2] < width \
            or (centery - height // 2) < 0 or (centerx - width // 2) < 0 \
            or (centery + height // 2) > img.shape[1] or (centerx + width // 2) > img.shape[2]:
        print("not satisfy")
        padding_margin = 50
        padding = [(0, 0), (padding_margin, padding_margin), (padding_margin, padding_margin)]
        img = np.pad(img, padding, mode='constant', constant_values=0)
        centery += padding_margin
        centerx += padding_margin

    return img[centerz - depth // 2:centerz + depth // 2, centery - height // 2:centery + height // 2,
           centerx - width // 2:centerx + width // 2]



def crop_start(img, startx, starty, startz, width, height, depth):
    return img[startz:startz + depth, starty:starty + height, startx:startx + width]


def save_volume(crop_3D_arr, boundingbox_path, _volume_info, name):
    crop_3D = sitk.GetImageFromArray(crop_3D_arr)
    crop_3D.SetOrigin(_volume_info[0])
    crop_3D.SetDirection(_volume_info[1])
    crop_3D.SetSpacing(_volume_info[2])
    sitk.WriteImage(crop_3D, os.path.join(boundingbox_path, name))


def all_data(nii_list):

    num = 0
    maxyy = 320
    maxxx = 448

    for i in tqdm(range(len(nii_list))):
        # mask和img名字一样
        masks_imgs_name = nii_list[i]

        print("\n")
        print(masks_imgs_name)

        # 读取 (Width, Height, Depth)
        masks_img = sitk.ReadImage(os.path.join(masks_path, masks_imgs_name))
        imgs_img = sitk.ReadImage(os.path.join(imgs_path, masks_imgs_name))

        # imgs_img的一些参数
        origin = imgs_img.GetOrigin()
        direction = imgs_img.GetDirection()
        spacing = imgs_img.GetSpacing()
        _volume_info = [origin, direction, spacing]

        # 转为Array (Depth, Width, Height)
        masks = sitk.GetArrayFromImage(masks_img)
        imgs = sitk.GetArrayFromImage(imgs_img)

        print("origin shape:", masks.shape, imgs.shape)
        # Osaka mask int16 and img int16
        # zheda mask float64 and img int16/int32
        print("origin dtype:", masks.dtype, imgs.dtype)
        print("origin masks max,min:", np.max(masks), np.min(masks))
        print("origin imgs max,min:", np.max(imgs), np.min(imgs))

    #     # 计算每个volume的boundingbox大小，并crop
    #     zz, yy, xx, centery, centerx = cul_bounding_box(masks)
    #     print("Depth, Height, Width, centery, centerx", zz, yy, xx, centery, centerx)
    #     if maxyy < yy:
    #         num += 1
    #         print("num", num)
    #         maxyy = yy
    #         print("maxyy, maxxx", maxyy, maxxx)
    #     if maxxx < xx:
    #         num += 1
    #         print("num", num)
    #         maxxx = xx
    #         print("maxyy, maxxx", maxyy, maxxx)

    # print("final maxyy, maxxx, num", maxyy, maxxx, num)

        # crop
        bbox = find_bounding_box(masks_img)
        print("bbox---startx, starty, startz, width, height, depth: ", bbox)
        centerx = bbox[0] + bbox[3] // 2
        centery = bbox[1] + bbox[4] // 2
        centerz = bbox[2] + bbox[5] // 2
        print("centerx, centery, centerz: ", centerx, centery, centerz)
        # width, height = 448, 320
        width, height = 560, 400
        print("bounding box width, height:", width, height)
        mask_crop_3D_arr = crop_center(masks, centerx, centery, centerz, width, height, bbox[5])
        img_crop_3D_arr = crop_center(imgs, centerx, centery, centerz, width, height, bbox[5])
        print("crop depth, height, width:", mask_crop_3D_arr.shape, img_crop_3D_arr.shape)
        print("crop masks max,min:", np.max(mask_crop_3D_arr), np.min(mask_crop_3D_arr))
        print("crop imgs max,min:", np.max(img_crop_3D_arr), np.min(img_crop_3D_arr))

        # save_volume(mask_crop_3D_arr, masks_boundingbox_path, _volume_info, masks_imgs_name)
        save_volume(img_crop_3D_arr, imgs_boundingbox_path, _volume_info, masks_imgs_name)


if __name__ == "__main__":
    nii_list = getNiiList(masks_path)
    # print(nii_list)
    # nii_list = ['TZC1-CT-1031958-2559474-20200212.nii.gz']
    all_data(nii_list)