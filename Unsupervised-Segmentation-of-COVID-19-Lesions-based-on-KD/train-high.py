import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import shutil
import time
from torchvision.models import resnet18
# from torchvision.models import mobilenet_v3_small
from models.resnet_backbone_ct import modified_resnet18
# from models.resnet_backbone import modified_resnet18
from base_networks import ConvBlock
from PIL import Image
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

#ct
# mean_train = [0.08336143, 0.08336143, 0.08336143]
# std_train = [0.14190406, 0.14190406, 0.14190406]


def data_transforms(input_size_x, input_size_y, mean_train=mean_train, std_train=std_train):
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size_y, input_size_x)),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
    return data_transforms

# def data_transforms_inv():
#     data_transforms_inv = transforms.Compose([transforms.Normalize(mean=list(-np.divide(mean_train, std_train)), std=list(np.divide(1, std_train)))])
#     return data_transforms_inv

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)


def SimilarityBlock(x, num_filter, out_filter, pool_filter_size, pool_stride, pool_padding):
        conv1 = ConvBlock(num_filter, out_filter, 1, 1, 0, activation='prelu', norm=None)
        conv1.to(device)
        maxpool = torch.nn.MaxPool2d(pool_filter_size, pool_stride, pool_padding)  # 8 4 2

        act = torch.nn.Sigmoid()

        # print("1---x.shape: ", x.shape)  # torch.Size([32, 64, 80, 112])

        x = conv1(x)
        x = maxpool(x)

        # print("2---x.shape: ", x.shape)

        batchSize = x.shape[0]
        channals = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]

        d_x = torch.reshape(x, (batchSize, w * h, channals))
        # print(d_x.shape)  # torch.Size([32, 8960, 64])
        d_x_t = torch.transpose(d_x, 1, 2)
        # print(d_x_t.shape)  # torch.Size([32, 64, 8960])

        s_x = torch.bmm(act(d_x), act(d_x_t))
        # print(s_x.shape, '11111111111111111111111111')  # torch.Size([32, 8960, 8960])

        return s_x



def cal_loss(fs_list, ft_list, criterion, criterion_l1):
    tot_f_loss = 0
    tot_s_f_loss = 0
    tot_loss = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        # print(i, fs.shape,ft.shape, "-----------")
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        # a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        # f_loss = (1/(w*h))*torch.sum(a_map)
        f_loss = (0.5/(w*h))*criterion(fs_norm, ft_norm)
        tot_f_loss += f_loss

        s_fs = SimilarityBlock(fs, 64*(2**i), 64*(2**i), 3, 2, 1)
        s_ft = SimilarityBlock(ft, 64*(2**i), 64*(2**i), 3, 2, 1)
        s_f_loss = criterion_l1(s_fs, s_ft)

        # s_f_loss = criterion_l1(fs, ft)

        tot_s_f_loss += s_f_loss

    tot_loss = tot_f_loss + lamda*tot_s_f_loss

    return tot_loss

def cal_anomaly_map(fs_list, ft_list, out_size_x, out_size_y):
    anomaly_map = np.ones([out_size_y, out_size_x])
    # print("anomaly_map shape: ", anomaly_map.shape)
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        a_map = torch.unsqueeze(a_map, dim=1)
        # print("1a_map: ", a_map.shape)
        a_map = F.interpolate(a_map, size=(out_size_y, out_size_x), mode='bilinear')
        # print("2a_map: ", a_map.shape)
        a_map = a_map[0,0,:,:].to('cpu').detach().numpy() # check
        a_map_list.append(a_map)
        anomaly_map *= a_map
    # for i in range(anomaly_map.shape[0]):
    #     anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
        # anomaly_map[i] = median_filter(anomaly_map[i], size=3)
    # # print("gaussian_filter")
    # print("median_filter")
    # print("no filter")
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    heatmap = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(heatmap) + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    

class STPM():
    def __init__(self):
        self.load_model()
        self.data_transform = data_transforms(input_size_x=input_size_x, input_size_y=input_size_y, mean_train=mean_train, std_train=std_train)

    def load_dataset(self):
        image_datasets = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=self.data_transform)
        # self.dataloaders = DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.dataloaders = DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
        dataset_sizes = {'train': len(image_datasets)}    
        print('Dataset size : Train set - {}'.format(dataset_sizes['train']))    

    def load_model(self):
        self.features_t = []
        self.features_s = []
        def hook_t(module, input, output):
            self.features_t.append(output)
        def hook_s(module, input, output):
            self.features_s.append(output)
        
        self.model_t = modified_resnet18(path=pretrain_path, pretrained=True).to(device)
        # self.model_t = modified_resnet18(pretrained=True).to(device)
        # self.model_t = resnet18(pretrained=True).to(device)
        self.model_t.layer1[-1].register_forward_hook(hook_t)
        self.model_t.layer2[-1].register_forward_hook(hook_t)
        self.model_t.layer3[-1].register_forward_hook(hook_t)

        self.model_s = modified_resnet18(path=None, pretrained=False).to(device)
        # self.model_s = modified_resnet18(pretrained=False).to(device)
        # self.model_s = resnet18(pretrained=False).to(device)
        self.model_s.layer1[-1].register_forward_hook(hook_s)
        self.model_s.layer2[-1].register_forward_hook(hook_s)
        self.model_s.layer3[-1].register_forward_hook(hook_s)

    def train(self):

        self.criterion = torch.nn.MSELoss(reduction='sum') # Does not use.
        self.criterion_l1 = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.model_s.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

        self.load_dataset()
        
        start_time = time.time()
        global_step = 0

        for epoch in range(num_epochs):
            print('-'*20)
            print('Time consumed : {}s'.format(time.time()-start_time))
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-'*20)
            self.model_t.eval()
            self.model_s.train()
            for idx, (batch, _) in enumerate(self.dataloaders): # batch loop
                global_step += 1
                batch = batch.to(device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    self.features_t = []
                    self.features_s = []
                    _ = self.model_t(batch)
                    _ = self.model_s(batch)
                    # get loss using features.
                    loss = cal_loss(self.features_s, self.features_t, self.criterion, self.criterion_l1)
                    loss.backward()
                    self.optimizer.step()

                if idx%2 == 0:
                    print('Epoch : {} | Loss : {:.4f}'.format(epoch, float(loss.data)))

        print('Total time consumed : {}'.format(time.time() - start_time))
        print('Train end.')
        if save_weight:
            print('Save weights.')
            torch.save(self.model_s.state_dict(), os.path.join(weight_save_path, 'model_s.pth'))
            torch.save(self.model_t.state_dict(), os.path.join(weight_save_path, 'model_t.pth'))
            # print("==============", self.model_t.state_dict())

    def test(self):
        print('Test phase start')
        try:
            self.model_s.load_state_dict(torch.load(glob.glob(weight_save_path+'/model_s.pth')[0]))
            # self.model_t.load_state_dict(torch.load(glob.glob(weight_save_path + '/model_t.pth')[0]))
        except:
            raise Exception('Check saved model path.')
        self.model_t.eval()
        self.model_s.eval()

        print("***************", dataset_path.split("/")[-1], "***************")
        test_path = os.path.join(dataset_path, 'test')
        gt_path = os.path.join(dataset_path, 'ground_truth')
        test_imgs_all = glob.glob(test_path + '/**/*.png', recursive=True)
        test_imgs = [i for i in test_imgs_all if "good" not in i]
        test_imgs_good = [i for i in test_imgs_all if "good" in i]
        gt_imgs = glob.glob(gt_path + '/**/*.png', recursive=True)
        test_imgs.sort()
        # print(test_imgs)
        gt_imgs.sort()
        # print(gt_imgs)
        gt_list_px_lvl = []
        gt_list_img_lvl = []
        pred_list_px_lvl = []
        pred_list_img_lvl = []
        start_time = time.time()
        print("Testset size : ", len(gt_imgs))        
        for i in range(len(test_imgs)):
            test_img_path = test_imgs[i]
            # print("test_img_path: ", test_img_path)  #  /home/ubuntu/users/wangyufeng/WYF_files/all-data/AD-mvtec/pill/test/color/021.png
            gt_img_path = gt_imgs[i]
            # print("gt_img_path: ", gt_img_path)  # #  /home/ubuntu/users/wangyufeng/WYF_files/all-data/AD-mvtec/pill/ground_truth/color/021_mask.png
            # print(os.path.split(test_img_path)[1], "192------------------------")  # 012.png
            # print(os.path.split(gt_img_path)[1], "193------------------------")  # 012_mask.png
            assert os.path.split(test_img_path)[1].split('.')[0] == os.path.split(gt_img_path)[1].split('.')[0], "Something wrong with test and ground truth pair!"  # ct
            # assert os.path.split(test_img_path)[1].split('.')[0] == os.path.split(gt_img_path)[1].split('_')[0], "Something wrong with test and ground truth pair!"  # MVTec
            defect_type = os.path.split(os.path.split(test_img_path)[0])[1]
            # print(defect_type, "196---------------------")  # color
            img_name = os.path.split(test_img_path)[1].split('.')[0]
            # print(img_name, "198--------------------")  # 021

            # ground truth
            gt_img_o = cv2.imread(gt_img_path,0)
            gt_img_o = cv2.resize(gt_img_o, (input_size_x, input_size_y))
            gt_list_px_lvl.extend(gt_img_o.ravel()//255)

            # load image
            test_img_o = cv2.imread(test_img_path)
            # print(test_img_o.shape, "215------------")  # (320,448,3)
            test_img_o = cv2.resize(test_img_o, (input_size_x, input_size_y))
            # print(test_img_o.shape, "217------------")  # (320,448,3)
            test_img = Image.fromarray(test_img_o)
            test_img = self.data_transform(test_img)
            # print(test_img.shape, "220-------------")  # torch.Size([3, 320, 448])
            test_img = torch.unsqueeze(test_img, 0).to(device)
            with torch.set_grad_enabled(False):
                self.features_t = []
                self.features_s = []
                _ = self.model_t(test_img)
                _ = self.model_s(test_img)
            # get anomaly map & each features
            anomaly_map, a_maps = cal_anomaly_map(self.features_s, self.features_t, out_size_x=input_size_x, out_size_y=input_size_y)
            pred_list_px_lvl.extend(anomaly_map.ravel())
            # pred_list_img_lvl.append(np.max(anomaly_map))
            # gt_list_img_lvl.append(1)

            if args.save_anomaly_map:

                # 0,1
                # print("anomaly_map.shape, unique:", anomaly_map.shape, np.unique(anomaly_map))
                anomaly_map_th = np.where(np.array(anomaly_map) > 0.009393300033609813, 1, 0)
                anomaly_map_th = anomaly_map_th*255

                # normalize anomaly amp
                anomaly_map_norm = min_max_norm(anomaly_map)
                anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)
                # 64x64 map
                am64 = min_max_norm(a_maps[0])
                # print(am64.shape, "====================", np.max(am64), np.min(am64))
                am64 = cvt2heatmap(am64*255)
                # 32x32 map
                am32 = min_max_norm(a_maps[1])
                am32 = cvt2heatmap(am32*255)
                # 16x16 map
                am16 = min_max_norm(a_maps[2])
                am16 = cvt2heatmap(am16*255)
                # anomaly map on image
                heatmap = cvt2heatmap(anomaly_map_norm*255)
                hm_on_img = heatmap_on_image(heatmap, test_img_o)

                # save images
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}.jpg'), test_img_o)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am64.jpg'), am64)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am32.jpg'), am32)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_am16.jpg'), am16)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_amap.jpg'), anomaly_map_norm_hm)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_amap_on_img.jpg'), hm_on_img)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_gt.jpg'), gt_img_o)
                cv2.imwrite(os.path.join(sample_path, f'{defect_type}_{img_name}_anomaly_map_th.jpg'), anomaly_map_th)
        
        # Test good image for image level score
        # for i in range(len(test_imgs_good)):
        #     test_img_path = test_imgs_good[i]
        #     defect_type = os.path.split(os.path.split(test_img_path)[0])[1]
        #     img_name = os.path.split(test_img_path)[1].split('.')[0]
        #
        #     # load image
        #     test_img_o = cv2.imread(test_img_path)
        #     test_img_o = cv2.resize(test_img_o, (input_size_x, input_size_y))
        #     test_img = Image.fromarray(test_img_o)
        #     test_img = self.data_transform(test_img)
        #     test_img = torch.unsqueeze(test_img, 0).to(device)
        #     with torch.set_grad_enabled(False):
        #         self.features_t = []
        #         self.features_s = []
        #         _ = self.model_t(test_img)
        #         _ = self.model_s(test_img)
        #     # get anomaly map & each features
        #     anomaly_map, a_maps = cal_anomaly_map(self.features_s, self.features_t, out_size_x=input_size_x, out_size_y=input_size_y)
        #     pred_list_img_lvl.append(np.max(anomaly_map))
        #     gt_list_img_lvl.append(0)
                    
        print('Total test time consumed : {}'.format(time.time() - start_time))
        # print("Total image-level roc-auc score :")
        # print(roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)

        self.gt_list_px_lvl, self.pred_list_px_lvl = gt_list_px_lvl, pred_list_px_lvl

    def evaluation(self):

        print(np.unique(self.gt_list_px_lvl), np.unique(self.pred_list_px_lvl))
        print("Total pixel-level auroc score :")
        print(roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl))

        print("Total pixel-level auprc score :")
        print(average_precision_score(self.gt_list_px_lvl, self.pred_list_px_lvl))

        precision, recall, thresholds = precision_recall_curve(self.gt_list_px_lvl, self.pred_list_px_lvl)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
        print("threshold: ", threshold)

        def dice(P, G):
            psum = np.sum(P.flatten())
            gsum = np.sum(G.flatten())
            pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
            score = (2 * pgsum) / (psum + gsum)
            return score

        print("Dice :")
        print(dice(np.where(np.array(self.pred_list_px_lvl) > threshold, 1, 0), np.array(self.gt_list_px_lvl)))



def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--dataset_path', default=r'./data/ct-onlylung-radiopaedia')  # ct-onlylung coronacases radiopaedia 20210702
    parser.add_argument('--num_epoch', default=2)  # 100 2
    parser.add_argument('--lr', default=0.01)  # 0.4 0.01
    parser.add_argument('--lamda', default=0.001)  # 0, 1, 0.001
    parser.add_argument('--batch_size', default=32)  # 32
    parser.add_argument('--input_size_x', default=448)
    parser.add_argument('--input_size_y', default=320)
    parser.add_argument('--pretrain_path', default=r'./models/weight-normal/netG_17.pth')
    # parser.add_argument('--pretrain_path', default=r'')
    # parser.add_argument('--project_path', default=r'/home/ubuntu/sdb/wangyufeng/STPM_anomaly_detection-main/Project_Train_Results/AD-covid19-icassp/affinity-lr0.01-epoch2-CTn17-15')
    # parser.add_argument('--project_path', default=r'/home/ubuntu/sdb/wangyufeng/STPM_anomaly_detection-main/Project_Train_Results/AD-covid19/covid19-modified_resnet18_22')
    parser.add_argument('--project_path', default=r'./Project_Train_Results/lamda0.001-lr0.01-epoch2-CTn17')
    parser.add_argument('--save_weight', default=True)
    parser.add_argument('--save_src_code', default=False)
    parser.add_argument('--save_anomaly_map', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    # print(torch.cuda.get_device_name(device))
    
    args = get_args()
    phase = args.phase
    dataset_path = args.dataset_path
    category = dataset_path.split('\\')[-1]
    num_epochs = args.num_epoch
    lr = args.lr
    lamda = args.lamda
    batch_size = args.batch_size
    save_weight = args.save_weight
    input_size_x = args.input_size_x  # weight
    input_size_y = args.input_size_y  # height
    save_src_code = args.save_src_code
    pretrain_path = args.pretrain_path
    project_path = args.project_path
    print("path-------------------------------", project_path)
    sample_path = os.path.join(project_path, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    weight_save_path = os.path.join(project_path, 'saved')
    if save_weight:
        os.makedirs(weight_save_path, exist_ok=True)
    if save_src_code:
        source_code_save_path = os.path.join(project_path, 'src')
        os.makedirs(source_code_save_path, exist_ok=True)
        copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README']) # copy source code
    

    stpm = STPM()
    if phase == 'train':
        stpm.train()
        stpm.test()
        stpm.evaluation()
    elif phase == 'test':
        stpm.test()
        # stpm.evaluation()
    else:
        print('Phase argument must be train or test.')
