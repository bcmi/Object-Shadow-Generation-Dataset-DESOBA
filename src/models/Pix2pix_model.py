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
from skimage.measure import compare_mse



import time
import torch
import itertools
from PIL import Image
from torch.autograd import Variable
import torchvision
import util.ssim as ssim








class Pix2pixModel(DistangleModel):
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
        self.loss_names = ['rescontruction']
        self.loss_names.append('D')
        self.loss_names.append('G_GAN')
        

        if self.isTrain:
            self.visual_names = ['shadowfree_img', 'shadow_img','instance_mask', 'shadow_mask', 'final']
        else:
            self.visual_names = ['final']




        

        self.model_names = ['MSP']
        self.model_names.append('D')
        

        opt.output_nc = 3
        self.netMSP = networks.define_G(4, 3, opt.ngf, 'unet_32', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, 5, opt.norm, False, opt.init_type,
                                              opt.init_gain, self.gpu_ids)
        self.netMSP.to(self.device)
        self.netD.to(self.device)
        

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.no_lsgan).to(self.device)
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_MSP = torch.optim.Adam(self.netMSP.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            


            self.optimizers.append(self.optimizer_MSP)
            
            self.optimizers.append(self.optimizer_D)
            

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
        if self.opt.residual:
            self.final_residual = self.netMSP(torch.cat([self.shadowfree_img, self.instance_mask],1))
            self.final = self.final_residual + self.shadowfree_img
        else:
            self.final = self.netMSP(torch.cat([self.shadowfree_img, self.instance_mask],1))
            ####shadowgan baselines
            # self.multiply = self.netMSP(torch.cat([self.shadowfree_img, self.instance_mask],1))
            # self.final = self.shadowfree_img * self.multiply
            # if self.opt.lambda_REF1:
            #     self.final_refine = self.netREF(self.final)



        
        diff = torch.abs(self.final - self.shadowfree_img)
        diff = diff * 2 - 1

        self.shadow_mask_predict = torch.zeros(self.shadow_mask.size()).cuda()
        if diff.size()[0] == 1:
            diff = torch.squeeze(diff)
            diff = torchvision.transforms.ToPILImage()(diff.detach().cpu())
            self.shadow_mask_predict = torchvision.transforms.Grayscale(num_output_channels=1)(diff)
            self.shadow_mask_predict = torchvision.transforms.ToTensor()(self.shadow_mask_predict).cuda()
            self.shadow_mask_predict = self.shadow_mask_predict*2 - 1
            self.shadow_mask_predict = self.shadow_mask_predict.unsqueeze(0)
        else:
            detach_diff = diff.detach().cpu()
            for i in range(int(diff.size()[0])):
                cu_diff = detach_diff[i]
                cu_diff = torchvision.transforms.ToPILImage()(cu_diff)
                cu_diff = torchvision.transforms.Grayscale(num_output_channels=1)(cu_diff)
                cu_diff = torchvision.transforms.ToTensor()(cu_diff).cuda()
                cu_diff = cu_diff * 2 - 1
                self.shadow_mask_predict[i] = cu_diff.unsqueeze(0)


        
        self.shadowmask_pred = self.shadow_mask_predict
        self.shadow_diff = torch.abs((self.shadowmask_pred/2+0.5) - (self.shadow_mask/2+0.5))*2 -1





    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        fake_final = self.final
        pred_fake = self.netD(fake_final.detach())

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_final = self.shadow_img
        pred_real = self.netD(real_final)

        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()



    def backward(self):
        #####GAN loss
        self.loss_rescontruction = self.MSELoss(self.final, self.shadow_img)
        self.loss = self.loss_rescontruction * self.opt.lambda_I1            
        pred_real = self.netD(self.final)
        self.loss_G_GAN = self.criterionGAN(pred_real, True)
        self.loss += self.loss_G_GAN * self.opt.lambda_GAN
        self.loss.backward()

    def optimize_parameters(self):
        if self.isTrain:
            self.forward()
            #####update discriminator
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights
            

            ######update generator
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_MSP.zero_grad()
            self.backward()
            self.optimizer_MSP.step()
        else:
            with torch.no_grad():
                self.forward()


    def get_current_visuals(self):
        t = time.time()
        nim = self.shadowfree_img.shape[0]
        visual_ret = OrderedDict()
        all = []
        for i in range(0, min(nim,10)):
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
        if self.opt.residual:
            self.final_residual = self.netMSP(torch.cat([self.shadowfree_img, self.instance_mask],1))
            self.final = self.final_residual + self.shadowfree_img
        else:
            self.final = self.netMSP(torch.cat([self.shadowfree_img, self.instance_mask],1))

