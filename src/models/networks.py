import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
###############################################################################
# Helper Functions
###############################################################################




import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
import sys
sys.path.append('../models/')
from torchvision.ops import RoIAlign, RoIPool
from util.boundding_box_utils import BoxCoder


def Bbox_To_Mask(rect, mask):
    for i in range(mask.size()[0]):
        mask[i,:,int(rect[i,0]):int(rect[i,2]), int(rect[i,1]):int(rect[i,3])] = 1
    # rect_mask = rect_mask*2 - 1
    return mask

def Bbox_To_Mask_2(rects, instance_mask):
    rect_mask = torch.zeros(instance_mask.size()).cuda()
    for i in range(len(rects)):
        rect = rects[i]
        for i in range(instance_mask.size()[0]):
            rect_mask[i,:,int(rect[i,0]):int(rect[i,2]), int(rect[i,1]):int(rect[i,3])] = 1
    rect_mask = rect_mask*2 - 1
    return rect_mask

def Bbox_To_Mask_1(rects, instance_mask, img_wh):
    rect_mask = torch.zeros(instance_mask.size()).cuda()
    for i in range(len(rects)):
        rect = rects[i] * img_wh
        for j in range(instance_mask.size()[0]):
            rect_mask[j,:,int(rect[j,0]):int(rect[j,2]), int(rect[j,1]):int(rect[j,3])] = 1
    rect_mask = rect_mask*2 - 1
    return rect_mask

def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale

def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5
    w_half *= scale
    h_half *= scale
    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp

import math
from torch.nn.modules.utils import _ntuple
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
                scale_factor is not None
                and isinstance(scale_factor, tuple)
                and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)

def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    # Need to work on the CPU, where fp16 isn't supported - cast to float to avoid this
    mask = mask.float()
    box = box.float()
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)
    mask = mask.expand((1, 1, -1, -1))
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(w, h), mode='bilinear', align_corners=False)

    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)
    im_coords = [x_0,y_0,x_1,y_1]
    mask_coords = [x_0 - box[0], y_0 - box[1], x_1 - box[0],  y_1 - box[1]]
    return im_coords, mask_coords, mask


def ROIMask_To_Img(masks, rects, instance_mask):
    im_mask = torch.zeros_like(instance_mask)
    for i in range(rects.size()[0]):
        for j in range(rects.size()[1]):
            rect = rects[i,j]
            mask = (masks[i,j] + 1)/2
            im_coord, mask_coord, new_mask = paste_mask_in_image(mask, rect, instance_mask.size()[-1], instance_mask.size()[-2], 0, 1)
            im_mask[i,:,im_coord[0]:im_coord[2],im_coord[1]:im_coord[3]] = new_mask[:,:,mask_coord[0]:mask_coord[2],mask_coord[1]:mask_coord[3]]
    return im_mask*2 -1








def Bbox_To_Mask_old(rect, instance_mask, img_wh):
    rect_mask = torch.zeros(instance_mask.size()).cuda()
    rect = rect * img_wh
    for i in range(instance_mask.size()[0]):
        rect_mask[i,:,int(rect[i,0]):int(rect[i,2]), int(rect[i,1]):int(rect[i,3])] = 1
    rect_mask = rect_mask*2 - 1
    return rect_mask

def Bbox_regression(shadow_mask, img_wh):
    rect = torch.zeros(shadow_mask.size()[0],5).cuda()
    rects = []
    for i in range(shadow_mask.size()[0]):
        valid_index = (shadow_mask[i,0] == 1).nonzero(as_tuple=False)
        if len(valid_index)< 1:
            x_left = 0
            x_right = 0
            y_top = 0
            y_bottom = 0
        else:
            x_left = torch.min(valid_index[:,0])
            x_right = torch.max(valid_index[:,0])
            y_top = torch.min(valid_index[:,1])
            y_bottom = torch.max(valid_index[:,1])
        rect[i,0] = 0
        rect[i,1] = x_left
        rect[i,2] = y_top
        rect[i,3] = x_right
        rect[i,4] = y_bottom
        rects.append(x_left)
        rects.append(y_bottom)
        rects.append(x_right)
        rects.append(y_top)
    rect = rect / img_wh
    rect_mask = Bbox_To_Mask_old(rect, shadow_mask, img_wh)
    return rect, rect_mask

#######unet attention
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_woCONECTION(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#######unet attention


def _get_kernel(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    """
        normalization
    :param in_:
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)


class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        gaussian_kernel = np.float32(_get_kernel(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)       # normalization
        new_attention = soft_attention.max(attention)
        return new_attention



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'shadow_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70000,90000,13200], gamma=0.3)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


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


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_vgg(num_input,num_classes,init_type='normal', init_gain=0.02, gpu_ids=[]):
    print(gpu_ids)
    from .vgg import create_vgg
    net = create_vgg(num_input,num_classes)
    net.to(gpu_ids[0])
    net = torch.nn.DataParallel(net,gpu_ids)
    init_weights(net,init_type,gain=init_gain)
    return net

class simple_CNN(nn.Module):
    def __init__(self,  num_input, num_output):
        super(simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_input, 32, kernel_size=7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 128)
        self.fc = nn.Linear(128, num_output)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = self.avgpool(x)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_12blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=12)

    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_32':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'SA':
        net = SA()
    elif netG == 'simpleCNN':
        net = simple_CNN(input_nc, output_nc)
    elif netG == 'RESNEXT':
        from .resnet import resnext101_32x8d
        net = resnext101_32x8d(pretrained=False,num_classes=output_nc,num_inputchannels=input_nc)
        if len(gpu_ids)>0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net,gpu_ids)
        return net
    elif netG == 'RESNEXT18':
        from .resnet import resnet18
        net = resnet18(pretrained=False,num_classes=output_nc,num_inputchannels=input_nc)
        if len(gpu_ids)>0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net,gpu_ids)
        return net

    elif netG == 'Res_twobranch9':
        net = ResnetGenerator_Twobranch(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Res_twobranch1':
        net = ResnetGenerator_Twobranch(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=1)
    elif netG == 'Res_twobranch3':
        net = ResnetGenerator_Twobranch(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
    elif netG == 'Res_twobranch6':
        net = ResnetGenerator_Twobranch(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'cc':
        net = ResnetGenerator_SelfATT(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'selfattention_resnet9':
        net = ResnetGenerator_SelfATT(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'unet_attention':
        net = Attention_UNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'unet_attention_multiply':
        net = Attention_UNet_multiply(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Attention_Twobranch_UNet':
        net = Attention_Twobranch_UNet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Attention_Twobranch_UNet_Noshare':
        net = Attention_Twobranch_UNet_Noshare(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Adain_Mask_Generator':
        # net = Adain_Mask_Generator(nf=ngf, input_dim_c=input_nc,input_dim_s=input_nc,output_dim=output_nc, nf_mlp=256,down_class=4,down_content=3,n_mlp_blks=3,n_res_blks=2,latent_dim=64)
        net = Adain_Mask_Generator(nf=ngf, input_dim_c=input_nc+1,input_dim_s=input_nc,output_dim=output_nc, nf_mlp=256,down_class=4,down_content=3,n_mlp_blks=3,n_res_blks=2,latent_dim=64)
    elif netG == 'Attention_Twobranch_UNet_Bbox':
        net = Attention_Twobranch_UNet_Bbox(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Attention_Twobranch_UNet_Noshare_bbox':
        net = Attention_Twobranch_UNet_Noshare_bbox(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Attention_Twobranch_UNet_Bbox_TwoDecoder':
        net = Attention_Twobranch_UNet_Bbox_TwoDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Attention_Twobranch_UNet_Noshare_bbox_TwoDecoder':
        net = Attention_Twobranch_UNet_Noshare_bbox_TwoDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Attention_UNet_MaskBox_Multitask':
        net = Attention_UNet_MaskBox_Multitask(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Attention_UNet_MaskBox_ROIMaskRCNN':
        net = Attention_UNet_MaskBox_ROIMaskRCNN(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'Attention_MaskBox_ROIMaskRCNN':
        net = Attention_MaskBox_ROIMaskRCNN(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'STN':
        net = STN_Network(input_channel=input_nc, ngf=ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'unet_32':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.fusion_conv = nn.Conv2d(in_channels =  2 *in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x_f, x_b_total):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x_f.size()
        # outs = []
        outs = torch.zeros(x_f.size()).cuda()
        for x_b in (x_b_total):
            proj_query  = self.query_conv(x_b).view(m_batchsize,-1,width*height).permute(0,2,1) # B X C X (N)
            proj_key =  self.key_conv(x_f).view(m_batchsize,-1,width*height) # B X C x (*W*H)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy) # B X (N) X (N)
            # print(torch.max(attention), torch.min(attention))
            proj_value = self.value_conv(x_b).view(m_batchsize,-1,width*height) # B X C X N
            out = torch.bmm(proj_value,attention.permute(0,2,1) )
            out = out.view(m_batchsize,C,width,height)
            outs = outs + out
        
        final_out = torch.cat([outs*self.gamma, x_f],1)
        final_out = self.fusion_conv(final_out)
        return final_out



#####abaltion study
class Attention_UNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(Attention_UNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True
        self.inc = DoubleConv(input_nc, ngf)
        self.down1 = Down(ngf, ngf*2)
        self.down2 = Down(ngf*2, ngf*2*2)
        self.down3 = Down(ngf*2*2, ngf*2*2*2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf*2*2*2, ngf*2*2*2*2 // factor)
        self.up1 = Up(ngf*2*2*2*2, ngf*2*2*2 // factor, self.bilinear)
        self.up2 = Up(ngf*2*2*2, ngf*2*2 // factor, self.bilinear)
        self.up3 = Up(ngf*2*2, ngf*2 // factor, self.bilinear)
        self.up4 = Up(ngf*2, ngf, self.bilinear)
        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()

        self.ATTnet = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet2 = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet3 = Self_Attn(ngf*4*2, 'relu')

    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask):
        fo_input = torch.cat([input_img, fo_mask], 1)
        bg_mask = ((bg_shadow_mask/2+0.5) + (bg_instance_mask/2+0.5)) * 2 -1
        bp_input = torch.cat([input_img, bg_pure_mask], 1)
        bos_input = torch.cat([input_img, bg_mask],1)
        
        ######foreground feature
        f1 = self.inc(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg pure feature
        p1 = self.inc(bp_input)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)
        ######bg object-shadow feature
        os1 = self.inc(bos_input)
        os2 = self.down1(os1)
        os3 = self.down2(os2)
        os4 = self.down3(os3)
        os5 = self.down4(os4)

        #####bottleneck attention
        #####non-local attention
        bottleneck_feature = self.ATTnet(f5, [p5, os5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5, os5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5, os5])

        ##### no background branch
        # bottleneck_feature = f5

        ##### local attention
        # bottleneck_feature = torch.cat([f5, p5, os5], 1)
        # bottleneck_feature = self.conv_concat(bottleneck_feature)




        #######upsample
        x = self.up1(bottleneck_feature, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        output = self.outc(x)
        output = self.out(output)
        return output


class Attention_UNet_MaskBox_Multitask(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(Attention_UNet_MaskBox_Multitask, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True

        # self.conv_concat = nn.Conv2d(in_channels = ngf*2*2*2*3 , out_channels = ngf*2*2*2 , kernel_size= 1)
        self.inc = DoubleConv(input_nc, ngf)
        self.down1 = Down(ngf, ngf*2)
        self.down2 = Down(ngf*2, ngf*2*2)
        self.down3 = Down(ngf*2*2, ngf*2*2*2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf*2*2*2, ngf*2*2*2*2 // factor)
        self.up1 = Up(ngf*2*2*2*2, ngf*2*2*2// factor, self.bilinear)
        self.up2 = Up(ngf*2*2*3, ngf*2*2 // factor, self.bilinear)
        self.up3 = Up(ngf*2*3, ngf*2 // factor, self.bilinear)
        self.up4 = Up(ngf*3, ngf, self.bilinear)
        self.outc = OutConv(ngf*2, self.output_nc)

        self.out = nn.Tanh()

        self.upbox1 = Up(ngf*2*2*2*2, ngf*2*2*2 // factor, self.bilinear)
        self.upbox2 = Up(ngf*2*2*2, ngf*2*2 // factor, self.bilinear)
        self.upbox3 = Up(ngf*2*2, ngf*2 // factor, self.bilinear)
        self.upbox4 = Up(ngf*2, ngf, self.bilinear)


        self.ATTnet = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet2 = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet3 = Self_Attn(ngf*4*2, 'relu')

        self.fc_b = nn.Linear(ngf*56*56, 4)
        self.roi = RoIAlign((56, 56), spatial_scale=256/256,sampling_ratio=2)

    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask, instance_rects, shadow_rects):
        # def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask):
        fo_input = torch.cat([input_img, fo_mask], 1)
        bp_input = torch.cat([input_img, bg_pure_mask], 1)
        bg_mask = ((bg_shadow_mask/2+0.5) + (bg_instance_mask/2+0.5)) * 2 -1
        bos_input = torch.cat([input_img, bg_mask],1)
        

        ######foreground feature
        f1 = self.inc(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg pure feature
        p1 = self.inc(bp_input)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)
        ######bg object-shadow feature
        os1 = self.inc(bos_input)
        os2 = self.down1(os1)
        os3 = self.down2(os2)
        os4 = self.down3(os3)
        os5 = self.down4(os4)

        #####bottleneck attention
        #####non-local attention
        bottleneck_feature = self.ATTnet(f5, [p5, os5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5, os5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5, os5])


        ####bounding box  decoder
        xbox1 = self.upbox1(bottleneck_feature,f4)
        xbox2 = self.upbox2(xbox1,f3)
        xbox3 = self.upbox3(xbox2,f2)
        xbox4 = self.upbox4(xbox3,f1)
        ### regression bbox  version 1 : w roi wo delta bbox
        ### redictly regressing (x_left,y_bottom,x_right,y_top)
        ### more foreground object
        shadow_bbox_mask_pred = torch.zeros(fo_mask.size()).cuda()
        rects_num = shadow_rects.size()[1]
        bboxs = torch.zeros(shadow_rects.size()).cuda()
        boxes = []
        for i in range(rects_num):
            rect = torch.zeros(fo_mask.size()[0],5).cuda()
            rect[:,1:] = instance_rects[:,i,:]
            box_features = self.roi(xbox4, rect)
            box_features = box_features.view(box_features.size(0), -1)
            box_delta = self.fc_b(box_features)
            box_features = box_features.view(box_features.size(0), -1)
            bbox = F.sigmoid(self.fc_b(box_features))
            bboxs[:,i] = bbox*input_img.size()[-1]
            boxes.append(bbox)
        shadow_bbox_mask_pred = Bbox_To_Mask_1(boxes, fo_mask, input_img.size()[2])


        #######upsample, adding the feature of bounding box
        x1 = self.up1(bottleneck_feature, f4)
        # aadding the feature of the bounding box decoder branch
        x1 = torch.cat([x1,xbox1],1)
        x2 = self.up2(x1, f3)
        x2 = torch.cat([x2,xbox2],1)
        x3 = self.up3(x2, f2)
        x3 = torch.cat([x3, xbox3],1)
        x4 = self.up4(x3, f1)
        x4 = torch.cat([x4, xbox4],1)
        output = self.outc(x4)
        output = self.out(output)
        # return output
        return bboxs, shadow_bbox_mask_pred,  output

class Attention_UNet_MaskBox_ROIMaskRCNN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(Attention_UNet_MaskBox_ROIMaskRCNN, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True

        self.inc = DoubleConv(input_nc, ngf)
        self.down1 = Down(ngf, ngf*2)
        self.down2 = Down(ngf*2, ngf*2*2)
        self.down3 = Down(ngf*2*2, ngf*2*2*2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf*2*2*2, ngf*2*2*2*2 // factor)
        self.up1 = Up(ngf*2*2*2*2, ngf*2*2*2// factor, self.bilinear)
        self.up2 = Up(ngf*2*2*2, ngf*2*2 // factor, self.bilinear)
        self.up3 = Up(ngf*2*2, ngf*2 // factor, self.bilinear)
        self.up4 = Up(ngf*2, ngf, self.bilinear)
        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()

        self.ATTnet = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet2 = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet3 = Self_Attn(ngf*4*2, 'relu')

        # bbox
        self.roi_b = RoIAlign((7, 7), spatial_scale=16/256,sampling_ratio=2)
        self.conv_b = nn.Sequential(
            nn.Conv2d(ngf*2*2*2, 1024, kernel_size=7, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
            # nn.ReLU(inplace=True)
            # nn.sigmoid()
        )
        self.fc_b = nn.Linear(1024, 4)


        # mask
        self.roi_m = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=2)
        self.conv_m = self.double_conv = nn.Sequential(
            nn.Conv2d(ngf*2*2*2, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,2,2,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256,1,1,1,0),
            OutConv(256, self.output_nc),
            nn.Tanh()
        )

    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask, instance_rects, shadow_rects):
        # def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask):
        fo_input = torch.cat([input_img, fo_mask], 1)
        bp_input = torch.cat([input_img, bg_pure_mask], 1)
        bg_mask = ((bg_shadow_mask/2+0.5) + (bg_instance_mask/2+0.5)) * 2 -1
        bos_input = torch.cat([input_img, bg_mask],1)

        ######foreground feature
        f1 = self.inc(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg pure feature
        p1 = self.inc(bp_input)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)
        ######bg object-shadow feature
        os1 = self.inc(bos_input)
        os2 = self.down1(os1)
        os3 = self.down2(os2)
        os4 = self.down3(os3)
        os5 = self.down4(os4)

        #####bottleneck attention
        #####non-local attention
        bottleneck_feature = self.ATTnet(f5, [p5, os5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5, os5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5, os5])


        #### ROI BBOX AND MASK
        shadow_bbox_mask_pred = torch.zeros(fo_mask.size()).cuda()
        rects_num = shadow_rects.size()[1]
        rects = torch.zeros(shadow_rects.size()).cuda()
        masks = torch.zeros(shadow_rects.size()[0], shadow_rects.size()[1],1,28,28).cuda()
        boxes = []
        for i in range(rects_num):
            rect = torch.zeros(fo_mask.size()[0],5).cuda()
            rect[:,1:] = instance_rects[:,i,:]
            # box
            box_feature = self.roi_b(bottleneck_feature, rect)
            bbox = self.conv_b(box_feature)
            bbox = bbox.view(bbox.size()[0],-1)
            bbox = F.sigmoid(self.fc_b(bbox))
            rects[:,i] = bbox*input_img.size()[-1]
            boxes.append(bbox)

            #predict mask : roi 14*14 -> 28*28
            mask_feature = self.roi_m(bottleneck_feature, rect)
            mask = self.conv_m(mask_feature)
            masks[:,i] = mask

        shadow_bbox_mask_pred = Bbox_To_Mask_1(boxes, fo_mask, input_img.size()[2])
        shadow_mask_pred = ROIMask_To_Img(masks, rects, fo_mask)



        #######upsample, adding the feature of bounding box
        x1 = self.up1(bottleneck_feature, f4)
        x2 = self.up2(x1, f3)
        x3 = self.up3(x2, f2)
        x4 = self.up4(x3, f1)
        output = self.outc(x4)
        output = self.out(output)
        # return output
        # shadow_mask_pred = output
        return rects, masks, shadow_bbox_mask_pred, shadow_mask_pred, output

class Attention_MaskBox_ROIMaskRCNN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(Attention_MaskBox_ROIMaskRCNN, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True

        self.inc = DoubleConv(input_nc, ngf)
        self.down1 = Down(ngf, ngf*2)
        self.down2 = Down(ngf*2, ngf*2*2)
        self.down3 = Down(ngf*2*2, ngf*2*2*2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf*2*2*2, ngf*2*2*2*2 // factor)



        self.ATTnet = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet2 = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet3 = Self_Attn(ngf*4*2, 'relu')

        # bbox
        self.roi_b = RoIAlign((7, 7), spatial_scale=16/256,sampling_ratio=2)
        self.conv_b = nn.Sequential(
            nn.Conv2d(ngf*2*2*2, 1024, kernel_size=7, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
            # nn.ReLU(inplace=True)
            # nn.sigmoid()
        )
        self.fc_b = nn.Linear(1024, 4)


        # mask
        self.roi_m = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=2)
        self.conv_m = self.double_conv = nn.Sequential(
            nn.Conv2d(ngf*2*2*2, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,2,2,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256,1,1,1,0),
            OutConv(256, self.output_nc),
            nn.Tanh()
        )

    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask, instance_rects, shadow_rects):
        # def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask):
        fo_input = torch.cat([input_img, fo_mask], 1)
        bp_input = torch.cat([input_img, bg_pure_mask], 1)
        bg_mask = ((bg_shadow_mask/2+0.5) + (bg_instance_mask/2+0.5)) * 2 -1
        bos_input = torch.cat([input_img, bg_mask],1)

        ######foreground feature
        f1 = self.inc(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg pure feature
        p1 = self.inc(bp_input)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)
        ######bg object-shadow feature
        os1 = self.inc(bos_input)
        os2 = self.down1(os1)
        os3 = self.down2(os2)
        os4 = self.down3(os3)
        os5 = self.down4(os4)

        #####bottleneck attention
        #####non-local attention
        bottleneck_feature = self.ATTnet(f5, [p5, os5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5, os5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5, os5])


        #### ROI BBOX AND MASK
        shadow_bbox_mask_pred = torch.zeros(fo_mask.size()).cuda()
        rects_num = shadow_rects.size()[1]
        rects = torch.zeros(shadow_rects.size()).cuda()
        masks = torch.zeros(shadow_rects.size()[0], shadow_rects.size()[1],1,28,28).cuda()
        boxes = []
        for i in range(rects_num):
            rect = torch.zeros(fo_mask.size()[0],5).cuda()
            rect[:,1:] = instance_rects[:,i,:]
            # box
            box_feature = self.roi_b(bottleneck_feature, rect)
            bbox = self.conv_b(box_feature)
            bbox = bbox.view(bbox.size()[0],-1)
            bbox = F.sigmoid(self.fc_b(bbox))
            rects[:,i] = bbox*input_img.size()[-1]
            boxes.append(bbox)

            #predict mask : roi 14*14 -> 28*28
            mask_feature = self.roi_m(bottleneck_feature, rect)
            mask = self.conv_m(mask_feature)
            masks[:,i] = mask

        shadow_bbox_mask_pred = Bbox_To_Mask_1(boxes, fo_mask, input_img.size()[2])
        shadow_mask_pred = ROIMask_To_Img(masks, rects, fo_mask)



        output = shadow_mask_pred
        return rects, masks, shadow_bbox_mask_pred, shadow_mask_pred, output


class Attention_UNet_multiply(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(Attention_UNet_multiply, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True

        # self.conv_concat = nn.Conv2d(in_channels = ngf*2*2*2*3 , out_channels = ngf*2*2*2 , kernel_size= 1)


        self.inc = DoubleConv(input_nc, ngf)
        self.down1 = Down(ngf, ngf*2)
        self.down2 = Down(ngf*2, ngf*2*2)
        self.down3 = Down(ngf*2*2, ngf*2*2*2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf*2*2*2, ngf*2*2*2*2 // factor)
        self.up1 = Up(ngf*2*2*2*2, ngf*2*2*2 // factor, self.bilinear)
        self.up2 = Up(ngf*2*2*2, ngf*2*2 // factor, self.bilinear)
        self.up3 = Up(ngf*2*2, ngf*2 // factor, self.bilinear)
        self.up4 = Up(ngf*2, ngf, self.bilinear)
        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()

        self.ATTnet = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet2 = Self_Attn(ngf*4*2, 'relu')
        self.ATTnet3 = Self_Attn(ngf*4*2, 'relu')

    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask):
        fo_input = input_img * fo_mask
        bp_input = input_img * bg_pure_mask
        bg_mask = ((bg_shadow_mask/2+0.5) + (bg_instance_mask/2+0.5)) * 2 -1
        bos_input = input_img * bg_mask
        ######foreground feature
        f1 = self.inc(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg pure feature
        p1 = self.inc(bp_input)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)
        ######bg object-shadow feature
        os1 = self.inc(bos_input)
        os2 = self.down1(os1)
        os3 = self.down2(os2)
        os4 = self.down3(os3)
        os5 = self.down4(os4)

        #####bottleneck attention
        #####non-local attention
        bottleneck_feature = self.ATTnet(f5, [p5, os5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5, os5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5, os5])

        ##### no background branch
        # bottleneck_feature = f5

        ##### local attention
        # bottleneck_feature = torch.cat([f5, p5, os5], 1)
        # bottleneck_feature = self.conv_concat(bottleneck_feature)




        #######upsample
        x = self.up1(bottleneck_feature, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        output = self.outc(x)
        output = self.out(output)
        return output


class Attention_Twobranch_UNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super(Attention_Twobranch_UNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True
        self.inc1 = DoubleConv(input_nc, ngf)
        self.inc2 = DoubleConv(input_nc+1, ngf)

        self.down1 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 2 * 2)
        self.down3 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.up1 = Up(ngf * 2 * 2 * 2 * 2, ngf * 2 * 2 * 2 // factor, self.bilinear)
        self.up2 = Up(ngf * 2 * 2 * 2, ngf * 2 * 2 // factor, self.bilinear)
        self.up3 = Up(ngf * 2 * 2, ngf * 2 // factor, self.bilinear)
        self.up4 = Up(ngf * 2, ngf, self.bilinear)
        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()

        self.ATTnet = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet2 = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet3 = Self_Attn(ngf * 4 * 2, 'relu')


    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask):
        fo_input = torch.cat([input_img, fo_mask], 1)
        # bp_input = torch.cat([input_img, bg_pure_mask], 1)
        bg_mask = ((bg_shadow_mask / 2 + 0.5) + (bg_instance_mask / 2 + 0.5)) * 2 - 1
        # bos_input = torch.cat([input_img, bg_mask], 1)
        b_input = torch.cat([input_img, bg_mask, bg_pure_mask], 1)
        ######foreground feature
        f1 = self.inc1(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg feature
        p1 = self.inc2(b_input)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)


        #####bottleneck attention
        # print(f5.size(), p5.size(), os5.size())
        bottleneck_feature = self.ATTnet(f5, [p5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5])

        #######upsample
        x = self.up1(bottleneck_feature, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        output = self.outc(x)
        output = self.out(output)
        return output




class Attention_Twobranch_UNet_Bbox(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super(Attention_Twobranch_UNet_Bbox, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True
        self.inc1 = DoubleConv(input_nc, ngf)
        # self.inc2 = DoubleConv(input_nc+1, ngf)

        self.down1 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 2 * 2)
        self.down3 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.fc = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor, 4)
        self.up1 = Up(ngf * 2 * 2 * 2 * 2, ngf * 2 * 2 * 2 // factor, self.bilinear)
        self.up2 = Up(ngf * 2 * 2 * 2, ngf * 2 * 2 // factor, self.bilinear)
        self.up3 = Up(ngf * 2 * 2, ngf * 2 // factor, self.bilinear)
        self.up4 = Up(ngf * 2, ngf, self.bilinear)
        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.outc_refine_1 = OutConv(ngf + 1 + 1 + 1, ngf)
        self.outc_refine_2 = OutConv(ngf, ngf)
        self.outc_refine_3 = OutConv(ngf, self.output_nc)

        self.ATTnet = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet2 = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet3 = Self_Attn(ngf * 4 * 2, 'relu')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.outc_refine_1 = OutConv(ngf + 1 + 1 + 1, ngf)
        self.outc_refine_2 = OutConv(ngf, ngf)
        self.outc_refine_3 = OutConv(ngf, self.output_nc)

        #### w roi
        ## encoder
        # self.fc_b = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor*7*7, 4)
        # self.fc_b = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor*14*14, 4)
        self.fc_b = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor*28*28, 4)
        # self.fc_b1 = nn.Linear(128, 4)
        ## decoder
        # self.fc_b = nn.Linear(ngf*14*14, 4)


        # encoder
        # before
        # self.roi = RoIAlign((7, 7), spatial_scale=16/256,sampling_ratio=2)
        # now
        self.roi = RoIAlign((28, 28), spatial_scale=16/256,sampling_ratio=2)
        # self.roi = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=2)
        # self.roi = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=0)
        # decoder
        # self.roi = RoIAlign((14, 14), spatial_scale=256/256,sampling_ratio=0)

        # ### w roi w delta bbox
        # nn.init.normal_(self.fc_b.weight, std=0.001)
        # nn.init.constant_(self.fc_b.bias, 0)
        # self.Boxcoder = BoxCoder(weights=(10,10,5,5))



    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask, instance_rects, shadow_rects):
        fo_input = torch.cat([input_img, fo_mask], 1)
        bg_mask = ((bg_shadow_mask / 2 + 0.5) + (bg_instance_mask / 2 + 0.5)) * 2 - 1
        b_input = torch.cat([input_img, bg_mask], 1)
        ######foreground feature
        f1 = self.inc1(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg feature
        p1 = self.inc1(b_input)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)


        #####bottleneck attention
        # print(f5.size(), p5.size(), os5.size())
        bottleneck_feature = self.ATTnet(f5, [p5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5])


        # #outputting shadow bbox
        # x = self.avgpool(bottleneck_feature)
        # x = x.view(input_img.size()[0],-1)
        # x = self.fc(x)
        # bbox = torch.sigmoid(x)
        # shadow_bbox_mask_pred = Bbox_To_Mask(bbox, fo_mask, input_img.size()[2])
        # #outputting shadow bbox

        ### regression bbox  version 1 : w roi wo delta bbox
        ### redictly regressing (x_left,y_bottom,x_right,y_top)
        ### more foreground object
        shadow_bbox_mask_pred = torch.zeros(fo_mask.size()).cuda()
        rects_num = shadow_rects.size()[1]
        bboxs = torch.zeros(shadow_rects.size()).cuda()
        boxes = []
        for i in range(rects_num):
            rect = torch.zeros(fo_mask.size()[0],5).cuda()
            rect[:,1:] = instance_rects[:,i,:]
            box_features = self.roi(bottleneck_feature, rect)
            box_features = box_features.view(box_features.size(0), -1)
            # box_delta = self.fc_b(box_features)
            box_features = box_features.view(box_features.size(0), -1)
            bbox = F.sigmoid(self.fc_b(box_features))
            bboxs[:,i] = bbox*input_img.size()[-1]
            boxes.append(bbox)
        shadow_bbox_mask_pred = Bbox_To_Mask_1(boxes, fo_mask, input_img.size()[2])



        #######upsample
        x = self.up1(bottleneck_feature, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        output = self.outc(x)
        output = self.out(output)
        # print(output.size())
        # print(shadow_bbox_mask_pred.size())
        # print(fo_mask.size())
        refine = self.outc_refine_1(torch.cat([x, output, shadow_bbox_mask_pred, fo_mask], 1))
        refine = self.outc_refine_2(refine)
        refine = self.outc_refine_3(refine)
        refine = self.out(refine)

        return bboxs, shadow_bbox_mask_pred, output, refine


class Attention_Twobranch_UNet_Bbox_TwoDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super(Attention_Twobranch_UNet_Bbox_TwoDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True
        self.inc1 = DoubleConv(input_nc, ngf)
        # self.inc2 = DoubleConv(input_nc+1, ngf)

        self.down1 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 2 * 2)
        self.down3 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.fc = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor, 4)
        self.up1 = Up(ngf * 2 * 2 * 2 * 2, ngf * 2 * 2 * 2 // factor, self.bilinear)
        self.up2 = Up(ngf * 2 * 2 * 2, ngf * 2 * 2 // factor, self.bilinear)
        self.up3 = Up(ngf * 2 * 2, ngf * 2 // factor, self.bilinear)
        self.up4 = Up(ngf * 2, ngf, self.bilinear)

        self.upbox1 = Up(ngf * 2 * 2 * 2 * 2, ngf * 2 * 2 * 2 // factor, self.bilinear)
        self.upbox2 = Up(ngf * 2 * 2 * 2, ngf * 2 * 2 // factor, self.bilinear)
        self.upbox3 = Up(ngf * 2 * 2, ngf * 2 // factor, self.bilinear)
        self.upbox4 = Up(ngf * 2, ngf, self.bilinear)


        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.outc_refine_1 = OutConv(ngf + 1 + 1 + 1, ngf)
        self.outc_refine_2 = OutConv(ngf, ngf)
        self.outc_refine_3 = OutConv(ngf, self.output_nc)

        self.ATTnet = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet2 = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet3 = Self_Attn(ngf * 4 * 2, 'relu')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.outc_refine_1 = OutConv(ngf + ngf + 1 + 1 + 1, ngf)
        self.outc_refine_2 = OutConv(ngf, ngf)
        self.outc_refine_3 = OutConv(ngf, self.output_nc)

        self.fc_b = nn.Linear(ngf*14*14, 4)


        # encoder
        # before
        # self.roi = RoIAlign((7, 7), spatial_scale=16/256,sampling_ratio=2)
        # now
        # self.roi = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=2)
        # self.roi = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=0)
        # decoder
        self.roi = RoIAlign((14, 14), spatial_scale=256/256,sampling_ratio=0)

        # ### w roi w delta bbox
        # nn.init.normal_(self.fc_b.weight, std=0.001)
        # nn.init.constant_(self.fc_b.bias, 0)
        # self.Boxcoder = BoxCoder(weights=(10,10,5,5))



    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask, instance_rects, shadow_rects, is_train):
        fo_input = torch.cat([input_img, fo_mask], 1)
        bg_mask = ((bg_shadow_mask / 2 + 0.5) + (bg_instance_mask / 2 + 0.5)) * 2 - 1
        b_input = torch.cat([input_img, bg_mask], 1)
        ######foreground feature
        f1 = self.inc1(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg feature
        p1 = self.inc1(b_input)
        p2 = self.down1(p1)
        p3 = self.down2(p2)
        p4 = self.down3(p3)
        p5 = self.down4(p4)

        #####bottleneck attention
        bottleneck_feature = self.ATTnet(f5, [p5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5])


        ### regression bbox  version 1 : w roi wo delta bbox
        ### redictly regressing (x_left,y_bottom,x_right,y_top)
        ### more foreground object
        box = self.upbox1(bottleneck_feature, f4)
        box = self.upbox2(box, f3)
        box = self.upbox3(box, f2)
        box = self.upbox4(box, f1)
        shadow_bbox_mask_pred = torch.zeros(fo_mask.size()).cuda()
        rects_num = shadow_rects.size()[1]
        bboxs = torch.zeros(shadow_rects.size()).cuda()
        boxes = []
        for i in range(rects_num):
            rect = torch.zeros(fo_mask.size()[0],5).cuda()
            rect[:,1:] = instance_rects[:,i,:]
            box_features = self.roi(box, rect)
            box_features = box_features.view(box_features.size(0), -1)
            box_delta = self.fc_b(box_features)
            box_features = box_features.view(box_features.size(0), -1)
            bbox = F.sigmoid(self.fc_b(box_features))
            bboxs[:,i] = bbox*input_img.size()[-1]
            boxes.append(bbox)
        shadow_bbox_mask_pred = Bbox_To_Mask_1(boxes, fo_mask, input_img.size()[2])


        #######upsample
        x = self.up1(bottleneck_feature, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        output = self.outc(x)
        output = self.out(output)

        # refinement module
        # randomly remove from predicted coarse shadow mask
        supervised_coarse_output = output
        if is_train:
            output = (output+1)/2
            for i in range(input_img.size()[0]):
                valid_index = (output[i,0] > 0.8).nonzero(as_tuple=False)
                # print('hh',valid_index)
                # print('gg', len(valid_index))
                if len(valid_index) > 1:
                    # x_left = torch.min(valid_index[:,0])
                    # x_right = torch.max(valid_index[:,0])
                    # y_top = torch.min(valid_index[:,1])
                    # y_bottom = torch.max(valid_index[:,1])
                    # w = x_right - x_left
                    # h = y_top - y_bottom
                    # w_ratio = torch.rand(1).cuda()
                    # h_ratio = torch.rand(1).cuda()
                    # output[i,0][x_left:x_left + (w_ratio*w).type(torch.int), y_bottom:y_bottom+(h_ratio*h).type(torch.int)] = 0
                    ratio = (torch.rand(1).cuda() * len(valid_index)).type(torch.int)
                    selected = valid_index[:ratio]
                    output[i,0,selected[:,0],selected[:,1]] = 0
            output = output * 2 -1


        refine = self.outc_refine_1(torch.cat([x, box, output, shadow_bbox_mask_pred, fo_mask], 1))
        refine = self.outc_refine_2(refine)
        refine = self.outc_refine_3(refine)
        refine = self.out(refine)

        return bboxs, shadow_bbox_mask_pred, supervised_coarse_output, output, refine

class Attention_Twobranch_UNet_Noshare_bbox(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super(Attention_Twobranch_UNet_Noshare_bbox, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True
        self.inc1 = DoubleConv(input_nc, ngf)
        self.inc11 = DoubleConv(input_nc, ngf)
        self.down1 = Down(ngf, ngf * 2)
        self.down11 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 2 * 2)
        self.down22 = Down(ngf * 2, ngf * 2 * 2)
        self.down3 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)
        self.down33 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.down44 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.fc = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor, 4)
        self.up1 = Up(ngf * 2 * 2 * 2 * 2, ngf * 2 * 2 * 2 // factor, self.bilinear)
        self.up2 = Up(ngf * 2 * 2 * 2, ngf * 2 * 2 // factor, self.bilinear)
        self.up3 = Up(ngf * 2 * 2, ngf * 2 // factor, self.bilinear)
        self.up4 = Up(ngf * 2, ngf, self.bilinear)
        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()



        self.ATTnet = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet2 = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet3 = Self_Attn(ngf * 4 * 2, 'relu')


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.outc_refine_1 = OutConv(ngf + 1 + 1 + 1, ngf)
        self.outc_refine_2 = OutConv(ngf, ngf)
        self.outc_refine_3 = OutConv(ngf, self.output_nc)

        #### w roi
        representation_dim = 128
        ## encoder
        # self.fc_b = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor*7*7, 4)
        self.fc_b = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor*14*14, 4)
        ## decoder
        # self.fc_b = nn.Linear(ngf*14*14, 4)


        # encoder
        # before
        # self.roi = RoIAlign((7, 7), spatial_scale=16/256,sampling_ratio=2)
        # now
        self.roi = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=2)
        # self.roi = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=0)
        # decoder
        # self.roi = RoIAlign((14, 14), spatial_scale=256/256,sampling_ratio=0)


        # ### w roi w delta bbox
        # nn.init.normal_(self.fc_b.weight, std=0.001)
        # nn.init.constant_(self.fc_b.bias, 0)
        # self.Boxcoder = BoxCoder(weights=(10,10,5,5))


    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask, instance_rects, shadow_rects):
        fo_input = torch.cat([input_img, fo_mask], 1)
        # bp_input = torch.cat([input_img, bg_pure_mask], 1)
        bg_mask = ((bg_shadow_mask / 2 + 0.5) + (bg_instance_mask / 2 + 0.5)) * 2 - 1
        # bos_input = torch.cat([input_img, bg_mask], 1)
        # b_input = torch.cat([input_img, bg_mask, bg_pure_mask], 1)
        b_input = torch.cat([input_img, bg_mask], 1)
        ######foreground feature
        f1 = self.inc1(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg feature
        p1 = self.inc11(b_input)
        p2 = self.down11(p1)
        p3 = self.down22(p2)
        p4 = self.down33(p3)
        p5 = self.down44(p4)

        #####bottleneck attention
        # print(f5.size(), p5.size(), os5.size())
        bottleneck_feature = self.ATTnet(f5, [p5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5])
        # print(bottleneck_feature.size()) #[1, 256, 16, 16]


        ### regression bbox  version 1 : w roi wo delta bbox
        ### redictly regressing (x_left,y_bottom,x_right,y_top)
        ### more foreground object
        shadow_bbox_mask_pred = torch.zeros(fo_mask.size()).cuda()
        rects_num = shadow_rects.size()[1]
        bboxs = torch.zeros(shadow_rects.size()).cuda()
        boxes = []
        for i in range(rects_num):
            rect = torch.zeros(fo_mask.size()[0],5).cuda()
            rect[:,1:] = instance_rects[:,i,:]
            box_features = self.roi(bottleneck_feature, rect)
            box_features = box_features.view(box_features.size(0), -1)
            box_delta = self.fc_b(box_features)
            box_features = box_features.view(box_features.size(0), -1)
            bbox = F.sigmoid(self.fc_b(box_features))
            bboxs[:,i] = bbox*input_img.size()[-1]
            boxes.append(bbox)
        shadow_bbox_mask_pred = Bbox_To_Mask_1(boxes, fo_mask, input_img.size()[2])

        # # ### regression bbox  version 1 : w roi w delta bbox,
        # # ### regressing (delta_x_center,delta_y_center,delta_w,delta_h)
        # rects_num = shadow_rects.size()[1]
        # shadow_bbox_mask_pred = torch.zeros(fo_mask.size()).cuda()
        # box_deltas = torch.zeros(shadow_rects.size()).cuda()
        # Boxes = torch.zeros(shadow_rects.size()).cuda()
        # boxes = []
        # for i in range(rects_num):
        #     rect = torch.zeros(fo_mask.size()[0],5).cuda()
        #     rect[:,1:] = instance_rects[:,i,:]
        #     box_features = self.roi(bottleneck_feature, rect)
        #     box_features = box_features.view(box_features.size(0), -1)
        #     box_delta = self.fc_b(box_features)
        #     box_deltas[:,i,:] = box_delta
        #     ###delta bbox: tx, ty, tw, th
        #     ## rect + box_delta (rect [N, 5])
        #     ## transform bbox to mask
        #     bbox = self.Boxcoder.decode(rel_codes=box_delta, boxes=rect)
        #     Boxes[:,i,:] = bbox
        #     boxes.append(bbox)
        #     # shadow_bbox_mask_pred += Bbox_To_Mask(bbox, shadow_bbox_mask_pred)
        # shadow_bbox_mask_pred = Bbox_To_Mask_2(boxes, fo_mask)
        # # shadow_bbox_mask_pred = shadow_bbox_mask_pred*2 -1


        #######upsample
        x = self.up1(bottleneck_feature, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)

        output = self.outc(x)
        output = self.out(output)
        refine = self.outc_refine_1(torch.cat([x, output, shadow_bbox_mask_pred, fo_mask], 1))
        refine = self.outc_refine_2(refine)
        refine = self.outc_refine_3(refine)
        refine = self.out(refine)




        return bboxs, shadow_bbox_mask_pred, output, refine
        # return box_deltas, shadow_bbox_mask_pred, output, refine
        # return Boxes, shadow_bbox_mask_pred, output, refine

class Attention_Twobranch_UNet_Noshare_bbox_TwoDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super(Attention_Twobranch_UNet_Noshare_bbox_TwoDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True
        self.inc1 = DoubleConv(input_nc, ngf)
        self.inc11 = DoubleConv(input_nc, ngf)
        self.down1 = Down(ngf, ngf * 2)
        self.down11 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 2 * 2)
        self.down22 = Down(ngf * 2, ngf * 2 * 2)
        self.down3 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)
        self.down33 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.down44 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.fc = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor, 4)
        self.up1 = Up(ngf * 2 * 2 * 2 * 2, ngf * 2 * 2 * 2 // factor, self.bilinear)
        self.up2 = Up(ngf * 2 * 2 * 2, ngf * 2 * 2 // factor, self.bilinear)
        self.up3 = Up(ngf * 2 * 2, ngf * 2 // factor, self.bilinear)
        self.up4 = Up(ngf * 2, ngf, self.bilinear)

        self.upbox1 = Up(ngf * 2 * 2 * 2 * 2, ngf * 2 * 2 * 2 // factor, self.bilinear)
        self.upbox2 = Up(ngf * 2 * 2 * 2, ngf * 2 * 2 // factor, self.bilinear)
        self.upbox3 = Up(ngf * 2 * 2, ngf * 2 // factor, self.bilinear)
        self.upbox4 = Up(ngf * 2, ngf, self.bilinear)

        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()



        self.ATTnet = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet2 = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet3 = Self_Attn(ngf * 4 * 2, 'relu')


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.outc_refine_1 = OutConv(ngf + ngf + 1 + 1 + 1, ngf)
        self.outc_refine_2 = OutConv(ngf, ngf)
        self.outc_refine_3 = OutConv(ngf, self.output_nc)

        #### w roi
        representation_dim = 128
        ## encoder
        # self.fc_b = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor*7*7, 4)
        # self.fc_b = nn.Linear(ngf * 2 * 2 * 2 * 2 // factor*14*14, 4)
        ## decoder
        self.fc_b = nn.Linear(ngf*14*14, 4)


        # encoder
        # before
        # self.roi = RoIAlign((7, 7), spatial_scale=16/256,sampling_ratio=2)
        # now
        # self.roi = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=2)
        # self.roi = RoIAlign((14, 14), spatial_scale=16/256,sampling_ratio=0)
        # decoder
        self.roi = RoIAlign((14, 14), spatial_scale=256/256,sampling_ratio=2)



    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask, instance_rects, shadow_rects, is_train):
        fo_input = torch.cat([input_img, fo_mask], 1)
        # bp_input = torch.cat([input_img, bg_pure_mask], 1)
        bg_mask = ((bg_shadow_mask / 2 + 0.5) + (bg_instance_mask / 2 + 0.5)) * 2 - 1
        # bos_input = torch.cat([input_img, bg_mask], 1)
        # b_input = torch.cat([input_img, bg_mask, bg_pure_mask], 1)
        b_input = torch.cat([input_img, bg_mask], 1)
        ######foreground feature
        f1 = self.inc1(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg feature
        p1 = self.inc11(b_input)
        p2 = self.down11(p1)
        p3 = self.down22(p2)
        p4 = self.down33(p3)
        p5 = self.down44(p4)

        #####bottleneck attention
        # print(f5.size(), p5.size(), os5.size())
        bottleneck_feature = self.ATTnet(f5, [p5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5])
        # print(bottleneck_feature.size()) #[1, 256, 16, 16]



        #######upsample
        x = self.up1(bottleneck_feature, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)

        xbox = self.upbox1(bottleneck_feature, f4)
        xbox = self.upbox2(xbox, f3)
        xbox = self.upbox3(xbox, f2)
        xbox = self.upbox4(xbox, f1)



        ### regression bbox  version 1 : w roi wo delta bbox
        ### redictly regressing (x_left,y_bottom,x_right,y_top)
        ### more foreground object
        shadow_bbox_mask_pred = torch.zeros(fo_mask.size()).cuda()
        rects_num = shadow_rects.size()[1]
        bboxs = torch.zeros(shadow_rects.size()).cuda()
        boxes = []
        for i in range(rects_num):
            rect = torch.zeros(fo_mask.size()[0],5).cuda()
            rect[:,1:] = instance_rects[:,i,:]
            box_features = self.roi(xbox, rect)
            box_features = box_features.view(box_features.size(0), -1)
            box_delta = self.fc_b(box_features)
            box_features = box_features.view(box_features.size(0), -1)
            bbox = F.sigmoid(self.fc_b(box_features))
            bboxs[:,i] = bbox*input_img.size()[-1]
            boxes.append(bbox)
        shadow_bbox_mask_pred = Bbox_To_Mask_1(boxes, fo_mask, input_img.size()[2])


        output = self.outc(x)
        output = self.out(output)


        # refinement module
        # randomly remove from predicted coarse shadow mask

        supervised_coarse_output = output
        if is_train:
            output = (output+1)/2
            for i in range(input_img.size()[0]):
                valid_index = (output[i,0] > 0.8).nonzero(as_tuple=False)
                if len(valid_index) > 1:
                    # x_left = torch.min(valid_index[:,0])
                    # x_right = torch.max(valid_index[:,0])
                    # y_top = torch.min(valid_index[:,1])
                    # y_bottom = torch.max(valid_index[:,1])
                    # w = x_right - x_left
                    # h = y_top - y_bottom
                    # w_ratio = torch.rand(1).cuda()
                    # h_ratio = torch.rand(1).cuda()
                    # output[i,0][x_left:x_left + (w_ratio*w).type(torch.int), y_bottom:y_bottom+(h_ratio*h).type(torch.int)] = 0
                    ratio = (torch.rand(1).cuda() * len(valid_index)).type(torch.int)
                    selected = valid_index[:ratio]
                    output[i,0,selected[:,0],selected[:,1]] = 0
            output = output * 2 -1

        refine = self.outc_refine_1(torch.cat([x, xbox, output, shadow_bbox_mask_pred, fo_mask], 1))
        refine = self.outc_refine_2(refine)
        refine = self.outc_refine_3(refine)
        refine = self.out(refine)


        return bboxs, shadow_bbox_mask_pred, supervised_coarse_output, output, refine
        # return box_deltas, shadow_bbox_mask_pred, output, refine
        # return Boxes, shadow_bbox_mask_pred, output, refine



class Attention_Twobranch_UNet_Noshare(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super(Attention_Twobranch_UNet_Noshare, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = True

        self.inc1 = DoubleConv(input_nc, ngf)
        self.inc11 = DoubleConv(input_nc+1, ngf)

        self.down1 = Down(ngf, ngf * 2)
        self.down11 = Down(ngf, ngf * 2)

        self.down2 = Down(ngf * 2, ngf * 2 * 2)
        self.down22 = Down(ngf * 2, ngf * 2 * 2)

        self.down3 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)
        self.down33 = Down(ngf * 2 * 2, ngf * 2 * 2 * 2)

        factor = 2 if self.bilinear else 1
        self.down4 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.down44 = Down(ngf * 2 * 2 * 2, ngf * 2 * 2 * 2 * 2 // factor)
        self.up1 = Up(ngf * 2 * 2 * 2 * 2, ngf * 2 * 2 * 2 // factor, self.bilinear)
        self.up2 = Up(ngf * 2 * 2 * 2, ngf * 2 * 2 // factor, self.bilinear)
        self.up3 = Up(ngf * 2 * 2, ngf * 2 // factor, self.bilinear)
        self.up4 = Up(ngf * 2, ngf, self.bilinear)
        self.outc = OutConv(ngf, self.output_nc)
        self.out = nn.Tanh()

        self.ATTnet = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet2 = Self_Attn(ngf * 4 * 2, 'relu')
        self.ATTnet3 = Self_Attn(ngf * 4 * 2, 'relu')



    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask):
        fo_input = torch.cat([input_img, fo_mask], 1)
        # bp_input = torch.cat([input_img, bg_pure_mask], 1)
        bg_mask = ((bg_shadow_mask / 2 + 0.5) + (bg_instance_mask / 2 + 0.5)) * 2 - 1
        # bos_input = torch.cat([input_img, bg_mask], 1)
        b_input = torch.cat([input_img, bg_mask, bg_pure_mask], 1)
        ######foreground feature
        f1 = self.inc1(fo_input)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)
        ######bg feature
        p1 = self.inc11(b_input)
        p2 = self.down11(p1)
        p3 = self.down22(p2)
        p4 = self.down33(p3)
        p5 = self.down44(p4)


        #####bottleneck attention
        # print(f5.size(), p5.size(), os5.size())
        bottleneck_feature = self.ATTnet(f5, [p5])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [p5])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [p5])



        #######upsample
        x = self.up1(bottleneck_feature, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        output = self.outc(x)
        output = self.out(output)
        return output





class ResnetGenerator_SelfATT(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_SelfATT, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        #####down sample
        # n_downsampling = 2
        n_downsampling = 4
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        ####bottle neck
        # model1 = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        ######upsampling
        model1 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                          kernel_size=3, stride=2,
                                          padding=1, output_padding=1,
                                          bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]

        #####final layer
        model1 += [nn.ReflectionPad2d(3)]
        model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model1 += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        self.ATTnet = Self_Attn(ngf*4*4, 'relu')
        self.ATTnet2 = Self_Attn(ngf*4*4, 'relu')
        self.ATTnet3 = Self_Attn(ngf*4*4, 'relu')
        # self.ATTnet4 = Self_Attn(ngf * 4, 'relu')
        # self.ATTnet5 = Self_Attn(ngf * 4, 'relu')

    def forward(self, input_img, fo_mask, bg_pure_mask, bg_shadow_mask, bg_instance_mask):
        fo_feature = self.model(torch.cat([input_img, fo_mask], 1))
        bg_pure_feature = self.model(torch.cat([input_img, bg_pure_mask], 1))

        ####background instance-shadow feature
        # bg_shadow_feature = self.model(torch.cat([input_img, bg_shadow_mask], 1))
        # bg_instance_feature = self.model(torch.cat([input_img, bg_instance_mask],1))
        # bottleneck_feature = self.ATTnet(fo_feature, [bg_pure_feature, bg_shadow_feature, bg_instance_feature])
        bg_mask = ((bg_shadow_mask/2+0.5) + (bg_instance_mask/2+0.5)) * 2 -1
        bg_instance_shadow_feature = self.model(torch.cat([input_img, bg_mask],1))
        # print('bottleneck feature size', fo_feature.size())
        bottleneck_feature = self.ATTnet(fo_feature, [bg_pure_feature, bg_instance_shadow_feature])
        bottleneck_feature = self.ATTnet2(bottleneck_feature, [bg_pure_feature, bg_instance_shadow_feature])
        bottleneck_feature = self.ATTnet3(bottleneck_feature, [bg_pure_feature, bg_instance_shadow_feature])
        # bottleneck_feature = self.ATTnet4(bottleneck_feature, [bg_pure_feature, bg_instance_shadow_feature])
        # bottleneck_feature = self.ATTnet5(bottleneck_feature, [bg_pure_feature, bg_instance_shadow_feature])
        output = self.model1(bottleneck_feature)
        return output






class ResnetGenerator_Twobranch(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_Twobranch, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 4
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        ######construct two branch
        # model1 = model
        # model2 = model
        model1, model2 = [], []

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                          kernel_size=3, stride=2,
                                          padding=1, output_padding=1,
                                          bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]

            model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                          kernel_size=3, stride=2,
                                          padding=1, output_padding=1,
                                          bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]


        model1 += [nn.ReflectionPad2d(3)]
        model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model1 += [nn.Tanh()]

        model2 += [nn.ReflectionPad2d(3)]
        model2 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model2 += [nn.Tanh()]


        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, input):
        x = self.model(input)
        return self.model1(x), self.model2(x)




# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)


        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)


        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)






# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        ####down sampling module
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        # self.net = [
        #     nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
        #     norm_layer(ndf * 2),
        #     nn.LeakyReLU(0.2, True),
        #
        #     nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
        #     norm_layer(ndf * 2),
        #     nn.LeakyReLU(0.2, True),
        #
        #     nn.Conv2d(ndf * 2 , ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
        #     norm_layer(ndf * 2),
        #     nn.LeakyReLU(0.2, True),
        #
        #     nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
        #     norm_layer(ndf * 2),
        #     nn.LeakyReLU(0.2, True),
        #
        #     nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        # print(self.net(input).size())
        return self.net(input)






class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, x_f, x_b_total):
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2



