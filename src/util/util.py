# _*_ coding:UTF-8 _*_
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def sdmkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2cvim(input_image, imtype=np.uint8,scale=None):
    # print('hhhhh',np.shape(input_image))

    if len(input_image.shape)<3: return None
    # if scale>0 and input_image.size()[1]==3:
    #     return tensor2im_logc(input_image, imtype=np.uint8,scale=scale)

    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.data[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy[image_numpy<0] = 0
    image_numpy[image_numpy>255] = 255
    return image_numpy.astype(imtype)


def tensor2imonechannel(input_image, imtype=np.uint8, scale=None):
    if len(input_image.shape) < 3: return None
    # if scale>0 and input_image.size()[1]==3:
    #     return tensor2im_logc(input_image, imtype=np.uint8,scale=scale)

    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.data[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy[image_numpy < 0] = 0
    # image_numpy[image_numpy > 255] = 255
    # return image_numpy.astype(imtype)
    return image_numpy



def tensor2im(input_image, imtype=np.uint8,scale=None):

    if len(input_image.shape)<3: return None
    # if scale>0 and input_image.size()[1]==3:
    #     return tensor2im_logc(input_image, imtype=np.uint8,scale=scale)

    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.data[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy[image_numpy<0] = 0
    image_numpy[image_numpy>255] = 255
    return image_numpy.astype(imtype)

def tensor2im_logc(image_tensor, imtype=np.uint8,scale=255):
    image_numpy = image_tensor.data[0].cpu().double().numpy()
    image_numpy = np.transpose(image_numpy,(1,2,0))
    image_numpy = (image_numpy+1) /2.0
    image_numpy = image_numpy * (np.log(scale+1))
    image_numpy = np.exp(image_numpy) -1
    image_numpy = image_numpy.astype(np.uint8)

    return image_numpy.astype(np.uint8)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def transparent_png(img,fg_mask):
    img = img.convert('RGBA')
    W, H = img.size
    # fg_mask = np.array(fg_mask)
    for h in range(H):   ###循环图片的每个像素点
        for l in range(W):
            if fg_mask.getpixel((l,h)) == 0:
                pixels = img.getpixel((l,h))
                if h==0 and l==0:
                    print('old',pixels)
                pixels = pixels[:-1] + (80,)
                if h==0 and l==0:
                    print('new',tuple(pixels))
                img.putpixel((l,h),pixels)
    img.save('/media/user/data/ShadowGeneration/HYShadowGeneration/SOBAMixFGAllData/Model_SelfAttention_GRESNEXT18_C32_Dpixel_lrD0.0002/SelfAttention_Illumination1_Residual0_ConditionD1_Llight0_Lpara1_Lshadowrecons10_Limagerecons10_Lgan0.1_Lstn0_Nwarp0_Lref1_Ltv0_Lganmask0/SelfAttention_GRESNEXT18_C32_Dpixel_lrD0.0002/web/images/0.png')
    # img = Image.open('/media/user/data/ShadowGeneration/HYShadowGeneration/SOBAMixFGAllData/Model_SelfAttention_GRESNEXT18_C32_Dpixel_lrD0.0002/SelfAttention_Illumination1_Residual0_ConditionD1_Llight0_Lpara1_Lshadowrecons10_Limagerecons10_Lgan0.1_Lstn0_Nwarp0_Lref1_Ltv0_Lganmask0/SelfAttention_GRESNEXT18_C32_Dpixel_lrD0.0002/web/images/0.png')
    # print(img.size())
    return  img