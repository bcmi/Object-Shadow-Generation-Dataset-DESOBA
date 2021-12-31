import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
from .distangle_model import DistangleModel
from PIL import ImageOps, Image
import cv2
import math
import sys
sys.path.append('../pytorch_ssim/')
import pytorch_ssim
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.nn import init

####stn
import time
import torch
import itertools
from PIL import Image
import torchvision


from maskshadowGAN_models_guided import Generator_F2S, Generator_S2F
from maskshadowGAN_models_guided import Discriminator
from maskshadowGAN_utils import ReplayBuffer
from maskshadowGAN_utils import LambdaLR
from maskshadowGAN_utils import weights_init_normal
# from maskshadowGAN_utils import mask_generator
from maskshadowGAN_utils import QueueMask

from sklearn.metrics import balanced_accuracy_score
from skimage.measure import compare_mse
import util.ssim as ssim





def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class MaskshadowGANModel(DistangleModel):
    def name(self):
        return 'Shadow Image Decomposition model ICCV19'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.add_argument('--wdataroot', default='None',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--use_our_mask', action='store_true')
        parser.add_argument('--mask_train', type=str, default=None)
        parser.add_argument('--mask_test', type=str, default=None)
        return parser


    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.opt.softattention = False

        self.loss_names = ['rescontruction' ,'GAN_A2B', 'GAN_B2A','recons_A','recons_B', 'id_shadow_image', 'id_shadowfree_image', \
                           'cycle_ABA', 'cycle_BAB' ,'D_B' ,'D_A', 'mask' ]


        if self.isTrain:
            self.visual_names = ['shadowfree_img', 'shadow_img', 'shadow_mask', 'fake_shadow']
        else:
            self.visual_names = ['fake_shadow']


        self.model_names = ['G_A2B','G_B2A', 'D_A', 'D_B']


        # self.netG_A2B = Generator_S2F(opt.input_nc, opt.output_nc)  # shadow to shadow_free
        # self.netG_B2A = Generator_F2S(opt.output_nc, opt.input_nc)  # shadow_free to shadow
        self.netG_A2B = Generator_F2S(opt.input_nc, opt.output_nc)  # shadow to shadow_free
        self.netG_B2A = Generator_S2F(opt.output_nc, opt.input_nc)  # shadow_free to shadow


        self.netD_A = Discriminator(opt.input_nc+1)
        self.netD_B = Discriminator(opt.output_nc+1)

        self.netG_A2B = init_net(self.netG_A2B, opt.init_type, opt.init_gain,self.gpu_ids)
        self.netG_B2A = init_net(self.netG_B2A, opt.init_type, opt.init_gain,self.gpu_ids)
        self.netD_A = init_net(self.netD_A, opt.init_type, opt.init_gain,self.gpu_ids)
        self.netD_B = init_net(self.netD_B, opt.init_type, opt.init_gain,self.gpu_ids)


        self.netG_A2B.to(self.device)
        self.netG_B2A.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)

        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)




        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.no_lsgan).to(self.device)
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G_A2B = torch.optim.Adam(self.netG_A2B.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_G_B2A = torch.optim.Adam(self.netG_B2A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers.append(self.optimizer_G_A2B)
            self.optimizers.append(self.optimizer_G_B2A)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)


    def set_input(self, input):
        self.shadow_img = input['A'].to(self.device)
        self.nim = self.shadow_img.shape[1]
        self.w = self.shadow_img.shape[2]
        self.h = self.shadow_img.shape[3]


        self.instance_mask = input['instancemask'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_mask_01 = (self.shadow_mask + 1) / 2
        self.shadow_mask = (self.shadow_mask_01 > 0.9).type(torch.float) * 2 - 1
        self.shadow_mask_3d = (self.shadow_mask > 0).type(torch.float).expand(self.shadow_img.shape)
        self.shadowfree_img = input['C'].to(self.device)
        self.shadow_param = input['param'].to(self.device).type(torch.float)
        self.bg_instance_mask = input['bg_instance'].to(self.device)
        self.bg_shadow_mask = input['bg_shadow'].to(self.device)

        self.bg_mask = ((self.bg_instance_mask/2+0.5) + (self.bg_shadow_mask/2+0.5))*2 - 1
        self.bg_pure_mask = (1 - (self.bg_mask/2+0.5) - (self.instance_mask/2+0.5) )*2 - 1







    def forward(self):
        ####[shadow image + mask] -> shadowfree image A->B
        #### shadowfree image -> shadow image B -> A
        mask_queue =  QueueMask(3000)


        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        #####identity
        self.nonmask = torch.zeros(self.shadow_mask.size())
        self.same_shadowfree_image  = self.netG_A2B(self.shadowfree_img ,  self.nonmask)
        self.same_shadow_image = self.netG_B2A(self.shadow_img)


        #####transfer
        self.fake_shadow = self.netG_B2A(self.shadowfree_img)

        ##calculated shadow mask
        # mask_queue.insert(mask_generator(self.fake_shadow, self.shadowfree_img))
        # self.mask_shadow = mask_queue.last_item()
        self.mask_shadow = self.shadow_mask

        self.fake_shadowfree = self.netG_A2B(self.shadow_img, self.shadow_mask)


        ###cyclegan
        self.recovered_shadowimage = self.netG_B2A(self.fake_shadowfree)
        self.recovered_shadowfree = self.netG_A2B(self.fake_shadow, self.shadow_mask)


        #####obtaining shadow difference
        diff = torch.abs(self.fake_shadow - self.shadowfree_img)
        diff = diff * 2 - 1
        diff = torch.squeeze(diff)
        # self.shadow_diff = torchvision.transforms.ToPILImage()(diff.cpu()).convert('Grey')
        diff = torchvision.transforms.ToPILImage()(diff.detach().cpu())
        self.shadow_mask_predict = torchvision.transforms.Grayscale(num_output_channels=1)(diff)
        self.shadow_mask_predict = torchvision.transforms.ToTensor()(self.shadow_mask_predict).cuda()
        self.shadow_mask_predict = self.shadow_mask_predict*2 - 1
        self.shadow_mask_predict = self.shadow_mask_predict.unsqueeze(0)


        self.shadow_diff = torch.abs((self.shadow_mask_predict/2+0.5) - (self.shadow_mask/2+0.5))*2 -1
        self.final = self.fake_shadow



    def backward_D_A(self):
        #DB2A_loss
        D_A_pred_fake = self.netD_A(torch.cat([self.fake_shadow.detach(), self.shadow_mask], 1) )
        self.loss_D_A_fake = self.criterionGAN(D_A_pred_fake, False)
        D_A_pred_real = self.netD_A(torch.cat([self.shadow_img, self.shadow_mask], 1))
        self.loss_D_A_real = self.criterionGAN(D_A_pred_real, True)
        self.loss_D_A = (self.loss_D_A_fake + self.loss_D_A_real) * 0.5
        self.loss_D_A.backward()

    def backward_D_B(self):
        #DA2B_loss
        D_B_pred_fake = self.netD_B(torch.cat([self.fake_shadowfree.detach(), self.shadow_mask],1))
        self.loss_D_B_fake = self.criterionGAN(D_B_pred_fake, False)
        D_B_pred_real = self.netD_B(torch.cat([self.shadowfree_img, self.shadow_mask],1))
        self.loss_D_B_real = self.criterionGAN(D_B_pred_real, True)
        self.loss_D_B = (self.loss_D_B_fake + self.loss_D_B_real) * 0.5
        self.loss_D_B.backward()




    def backward(self):
        ######generator
        # mask recons
        self.loss_mask = self.MSELoss(self.shadow_mask, self.mask_shadow)
        #identity recons
        self.loss_id_shadow_image = self.criterionL1(self.same_shadow_image, self.shadow_img)
        self.loss_id_shadowfree_image = self.criterionL1(self.same_shadowfree_image, self.shadowfree_img)

        #reconstruction loss
        self.loss_recons_A = self.MSELoss(self.shadow_img, self.fake_shadow)
        self.loss_recons_B = self.MSELoss(self.shadowfree_img, self.fake_shadowfree)

        #GA2B_loss
        pred_fake_shadowfree = self.netD_B(torch.cat([self.fake_shadowfree, self.shadow_mask],1))
        self.loss_GAN_A2B = self.criterionGAN(pred_fake_shadowfree, True)
        #GB2A_loss
        pred_fake_shadow = self.netD_A(torch.cat([self.fake_shadow, self.shadow_mask],1))
        self.loss_GAN_B2A = self.criterionGAN(pred_fake_shadow, True)

        #cycleloss B2A
        self.loss_cycle_ABA = self.MSELoss(self.recovered_shadowimage, self.shadow_img)
        #cycleloss A2B
        self.loss_cycle_BAB = self.MSELoss(self.recovered_shadowfree, self.shadowfree_img)

        self.loss = self.opt.lambda_GAN*(self.loss_GAN_A2B + self.loss_GAN_B2A) + self.opt.lambda_I1 * ( \
                    self.loss_recons_A + self.loss_recons_B + self.loss_id_shadow_image + self.loss_id_shadowfree_image +  self.loss_cycle_ABA + self.loss_cycle_BAB) + \
                    self.loss_mask * 0 #self.opt.lambda_M1
        self.loss.backward()

    def optimize_parameters(self):
        if self.isTrain:
            self.forward()

            #####update discriminator
            self.set_requires_grad(self.netD_A, True)  # enable backprop for D
            self.optimizer_D_A.zero_grad()  # set D's gradients to zero
            self.backward_D_A()  # calculate gradients for D
            self.optimizer_D_A.step()  # update D's weights


            #####update discriminator
            self.set_requires_grad(self.netD_B, True)  # enable backprop for D
            self.optimizer_D_B.zero_grad()  # set D's gradients to zero
            self.backward_D_B()  # calculate gradients for D
            self.optimizer_D_B.step()  # update D's weights

            ######update generator
            self.set_requires_grad(self.netD_A, False)  # D requires no gradients when optimizing G
            self.set_requires_grad(self.netD_B, False)  # D_mask requires no gradients when optimizing G
            self.optimizer_G_A2B.zero_grad()
            self.optimizer_G_B2A.zero_grad()
            self.backward()
            self.optimizer_G_A2B.step()
            self.optimizer_G_B2A.step()

        else:
            with torch.no_grad():
                self.forward()

    

    def get_current_visuals(self):
        t = time.time()
        nim = self.shadowfree_img.shape[0]
        visual_ret = OrderedDict()
        all = []
        for i in range(0, min(nim,2)):
            row = []
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self, name):
                        im = util.tensor2im(getattr(self, name).data[i:i + 1, :, :, :])
                        row.append(im)

            row = tuple(row)
            row = np.hstack(row)
            if hasattr(self, 'isreal'):
                if self.isreal[i] == 0:
                    row = ImageOps.crop(Image.fromarray(row), border=5)
                    row = ImageOps.expand(row, border=5, fill=(0, 200, 0))
                    row = np.asarray(row)
            all.append(row)
        all = tuple(all)

        if len(all) > 0:
            allim = np.vstack(all)
            return len(all), OrderedDict([(self.opt.name, allim)])
        else:
            return len(all), None


    def get_current_errors(self):
        #####MSE error
        RMSE = []
        shadowRMSE = []
        SSIM = []
        shadowSSIM = []

        t = time.time()
        nim = self.shadowfree_img.shape[0]
        visual_ret = OrderedDict()
        all = []
        for i in range(nim):
            gt = util.tensor2im(getattr(self, 'shadow_img').data[i:i + 1, :, :, :]).astype(np.float32)
            prediction = util.tensor2im(getattr(self, 'final').data[i:i + 1, :, :, :]).astype(np.float32)
            mask = util.tensor2imonechannel(getattr(self, 'shadow_mask').data[i:i + 1, :, :, :])
         
            RMSE.append(math.sqrt(compare_mse(gt, prediction)))
            shadowRMSE.append(math.sqrt(compare_mse(gt*(mask/255), prediction*(mask/255))*256*256/np.sum(mask/255)))
            
            gt_tensor = (getattr(self, 'shadow_img').data[i:i + 1, :, :, :]/2 + 0.5) * 255
            prediction_tensor = (getattr(self, 'final').data[i:i + 1, :, :, :]/2 + 0.5) * 255
            mask_tensor = (getattr(self, 'shadow_mask').data[i:i + 1, :, :, :]/2 + 0.5)
            SSIM.append(pytorch_ssim.ssim(gt_tensor, prediction_tensor, window_size = 11, size_average = True))
            shadowSSIM.append(ssim.ssim(gt_tensor, prediction_tensor,mask=mask_tensor))

        return RMSE,shadowRMSE, SSIM, shadowSSIM



    def prediction(self):
        self.fake_shadow = self.netG_B2A(self.shadowfree_img)
        self.final = self.fake_shadow

