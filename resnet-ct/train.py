"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64

train
python train.py --model ganomaly --dataroot /data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/train_median_3_patch --batchsize (64/128) --isize 64 --nz 512 --ngf 64 --ndf 64 --dataset lung --display --split train
python train.py --model ganomaly --dataroot /data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/train_median_3_patch --batchsize 64 --isize 64 --nz 512 --ngf 64 --ndf 64 --dataset lung --display --split train

python train.py --model ganomaly-normal --dataroot /data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data/normal_img_normalization_patch --batchsize 256 --isize 64 --nz 512 --ngf 64 --ndf 64 --dataset lung --display --split train --env ganomaly-normal
python train.py --model resnet18 --dataroot /data3T_1/wangyufeng/WYF_Files/Anomaly_detection/all-data-png/ct-onlylung-20210702-train-test/ --batchsize 16 --isize 64 --isize_x 448 --isize_y 320 --nz 512 --ngf 64 --ndf 64 --dataset lung --display --split train --env resnet18 --gpu_ids 1 --display_port 8097

test
python train.py --model ganomaly --dataroot /data3T_1/caoxiao/data/Lung/anomaly --batchsize 1 --isize 512 --nz 512 --ngf 64 --ndf 64 --dataset lung --save_test_images --load_weights --epoch --split test
python train.py --model CAE --dataroot /data3T_1/wangyufeng/WYF_Files/classification_wyf/data/Lung/anomaly/COVID-19/ann_COVID19 --batchsize 1 --isize 512 --nz 512 --ngf 64 --ndf 64 --dataset lung --save_test_images --load_weights --epoch 30 --split test
python train.py --model GANOMAlY-MOREDATA --dataroot /data3T_1/wangyufeng/WYF_Files/classification_wyf/data/Lung/anomaly/COVID-19/ann_COVID19 --batchsize 1 --isize 512 --nz 512 --ngf 64 --ndf 64 --dataset lung --save_test_images --load_weights --epoch 50 --split test
python train.py --model ganomaly-onlylung-nomedian --dataroot /data3T_1/wangyufeng/WYF_Files/classification_wyf/data/Lung/anomaly/COVID-19/ann_COVID19 --batchsize 1 --isize 512 --nz 512 --ngf 64 --ndf 64 --dataset lung --save_test_images --load_weights --epoch 50 --split test

"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Resnet

##
def train():
    """ Training
    """
    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Resnet(opt, dataloader)
    ##
    # TRAIN MODEL
    if opt.split == 'train':
        model.train()
    else:
        model.test()

if __name__ == '__main__':
    train()
