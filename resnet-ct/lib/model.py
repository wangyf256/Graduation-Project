"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from PIL import Image
from lib.resnet18_34_ct import ctResNet, weights_init
from lib.visualizer import Visualizer


class BaseModel():
    """ Base Model for ganomaly
    """

    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device('cuda:{}'.format(
            self.opt.gpu_ids[0]) if self.opt.device != 'cpu' else "cpu")

    ##
    def set_input(self, input: torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())
            self.names = input[2]

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """ Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            # ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            # ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item())])
        # ('err_g_enc', self.err_g_enc.item())])
        # ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[3].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch, is_best=False):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(
            self.opt.outf,
            self.opt.name,
            'train',
            'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if not is_best:
            torch.save({'epoch': epoch + 1,
                        'state_dict': self.netg.state_dict()},
                       '%s/netG_%d.pth' % (weight_dir,
                                           epoch + 1))
        else:
            torch.save({'epoch': epoch + 1,
                        'state_dict': self.netg.state_dict()},
                       '%s/netG_best.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(
            self.dataloader['train'],
            leave=False,
            total=len(
                self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / \
                        len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(
                        self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                # print(reals.shape, fakes.shape, fixed.shape, "----------------")
                self.visualizer.save_current_images(
                    self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" %
              (self.name, self.epoch + 1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            # res = self.test()
            # if res['AUC'] > best_auc:
            #     best_auc = res['AUC']
            #     self.save_weights(self.epoch, is_best=True)
            # else:
            self.save_weights(self.epoch, is_best=False)
            # self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                # path = "./output/{}/{}/train/weights/netG_{}.pth".format(self.name.lower(), self.opt.dataset, self.opt.epoch)
                path = "./output/{}/{}/train/weights/netG_{}.pth".format(
                    self.opt.model.lower(), self.opt.dataset, self.opt.epoch)
                print(path, "+" * 100)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            res_dict = {}
            res_dir = os.path.join(self.opt.outf, self.opt.name, 'test', 'res')
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)
            for i, data in enumerate(tqdm(self.dataloader['test'], 0)):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:

                    dst = os.path.join(
                        self.opt.outf, self.opt.name, 'test', 'images')
                    real_dst = os.path.join(
                        self.opt.outf, self.opt.name, 'test', 'real_images')
                    dst_error = os.path.join(
                        self.opt.outf, self.opt.name, 'test', 'errors')

                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    if not os.path.isdir(dst_error):
                        os.makedirs(dst_error)
                    if not os.path.isdir(real_dst):
                        os.makedirs(real_dst)
                    reals, fakes, _ = self.get_current_images()
                    for real, fake, name, e in zip(
                            reals, fakes, self.names, error):
                        # real = np.squeeze(real.detach().cpu().numpy())
                        # fake = np.squeeze(fake.detach().cpu().numpy())
                        # res = abs(real - fake)
                        # res = (res - np.min(res)) / (np.max(res) - np.min(res))
                        # res_dict[name] = res
                        fake = fake.clone()  # avoid modifying tensor in-place
                        real = real.clone()

                        def norm_ip(img, min, max):
                            img.clamp_(min=min, max=max)
                            img.add_(-min).div_(max - min + 1e-5)

                        def norm_range(t, range=None):
                            # t.add_(1).div(2)
                            if range is not None:
                                norm_ip(t, range[0], range[1])
                            else:
                                norm_ip(t, float(t.min()), float(t.max()))

                        res_img = real - fake
                        norm_range(res_img)
                        res_img = np.squeeze(
                            res_img.permute(
                                1, 2, 0).cpu().numpy())
                        res_img[res_img >= 0.7] = 255
                        res_img[res_img < 0.7] = 0
                        res_img = np.asarray(res_img, dtype=np.uint8)
                        res_img = Image.fromarray(res_img)
                        res_img.save(os.path.join(dst_error, name))

                        fake.clamp_(min=-1, max=1)
                        norm_range(fake)

                        # fake.add_(1).div_(2)
                        # print('fake:', fake.min(), fake.max())
                        fake = np.squeeze(
                            fake.mul(255).clamp(
                                0, 255).byte().permute(
                                1, 2, 0).cpu().numpy())
                        fake = Image.fromarray(fake)
                        fake.save(os.path.join(dst, name))

                        norm_range(real)
                        real = np.squeeze(
                            real.mul(255).clamp(
                                0, 255).byte().permute(
                                1, 2, 0).cpu().numpy())
                        real = Image.fromarray(real)
                        real.save(os.path.join(real_dst, name))

            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)



##
class Resnet(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self):
        return 'Resnet'

    def __init__(self, opt, dataloader):
        super(Resnet, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = ctResNet().to(self.device)
        self.netg.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(
                os.path.join(
                    self.opt.resume,
                    'netG.pth'))['epoch']
            self.netg.load_state_dict(
                torch.load(
                    os.path.join(
                        self.opt.resume,
                        'netG.pth'))['state_dict'])
            self.netd.load_state_dict(
                torch.load(
                    os.path.join(
                        self.opt.resume,
                        'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_con = nn.MSELoss()
        ##
        # Initialize input tensors.
        self.input = torch.empty(
            size=(
                self.opt.batchsize,
                3,
                self.opt.isize_y,
                self.opt.isize_x),
            dtype=torch.float32,
            device=self.device)
        self.label = torch.empty(
            size=(
                self.opt.batchsize,
            ),
            dtype=torch.float32,
            device=self.device)
        self.gt = torch.empty(
            size=(
                opt.batchsize,
            ),
            dtype=torch.long,
            device=self.device)
        self.fixed_input = torch.empty(
            size=(
                self.opt.batchsize,
                3,
                self.opt.isize_y,
                self.opt.isize_x),
            dtype=torch.float32,
            device=self.device)
        self.real_label = torch.ones(
            size=(
                self.opt.batchsize,
            ),
            dtype=torch.float32,
            device=self.device)
        self.fake_label = torch.zeros(
            size=(
                self.opt.batchsize,
            ),
            dtype=torch.float32,
            device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.optimizer_g = optim.Adam(
                self.netg.parameters(), lr=self.opt.lr, betas=(
                    self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        _, _, _, self.fake = self.netg(self.input)


    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        # print(self.fake.shape, self.input.shape, "+" * 100)  # torch.Size([16, 1, 320, 448]) torch.Size([16, 1, 320, 448])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g = self.err_g_con * self.opt.w_con
        self.err_g.backward(retain_graph=True)



    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()
