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


####stn
sys.path.append('../tps_stn_pytorch/')
import shadow_stn_model
import time
import torch
import itertools
from PIL import Image
from grid_sample import grid_sample
from torch.autograd import Variable
from tps_grid_gen import TPSGridGen
import torchvision
import util.ssim as ssim







def metrics(prediction_mask, gt_mask, step_nums):
    predicted_mask = prediction_mask.copy()
    mask = gt_mask

    # print('hhh',np.max(predicted_mask),  np.min(predicted_mask))

    # step = int((np.max(predicted_mask) - np.min(predicted_mask))/step_nums)

    step = int(256/step_nums)


    acc_final = 0
    acc_finals = []
    acc_mask = np.zeros(np.shape(predicted_mask))

    iou_final = 0
    iou_finals = []

    ber_final = 1
    ber_final_shadow = 1
    ber_finals = []
    ber_final_shadows = []

    for i in range(step_nums):
        predicted_mask = prediction_mask.copy()
        predicted_mask[predicted_mask >= (i + 1) * step - 1] = 255
        predicted_mask[predicted_mask != 255] = 0

        ####acc calculation
        acc_mask[predicted_mask==mask] = 1
        acc_mask[predicted_mask!=mask] = 0
        acc = np.sum(acc_mask) / int(np.shape(predicted_mask)[0]*np.shape(predicted_mask)[1]) * 100
        if acc > acc_final:
            acc_final = acc
        acc_finals.append(acc)

        ####iou calculation
        total = predicted_mask/255 + mask/255
        zeros0 = np.zeros(np.shape(predicted_mask))
        zeros0[total == 2] = 1
        inter = np.sum(zeros0)
        zeros00 = np.zeros(np.shape(predicted_mask))
        zeros00[predicted_mask == 255] = 1
        predict_mask_area = np.sum(zeros00)
        zeros11 = np.zeros(np.shape(predicted_mask))
        zeros11[mask == 255] = 1
        gt_mask_area = np.sum(zeros11)
        mask_iou_value = inter / (predict_mask_area + gt_mask_area - inter)
        if mask_iou_value > iou_final:
            iou_final = mask_iou_value
        iou_finals.append(iou_final)

        ######ber calculation
        zeros1 = np.zeros(np.shape(predicted_mask))
        zeros2 = np.zeros(np.shape(predicted_mask))
        zeros3 = np.zeros(np.shape(predicted_mask))
        zeros4 = np.zeros(np.shape(predicted_mask))
        ####[1,1]
        zeros1[total == 2] = 1
        ####[0,0]
        zeros2[total == 0] = 1
        ####[1,0]
        zeros3[(predicted_mask/255 - mask/255) == 1] = 1
        ####[0,1]
        zeros4[(predicted_mask/255 - mask/255) == -1] = 1
        TP = np.sum(zeros1)
        TN = np.sum(zeros2)
        FP = np.sum(zeros3)
        FN = np.sum(zeros4)
        TP_shadow = np.sum(zeros1 * (mask/255))
        TN_shadow = np.sum(zeros2 * (mask/255))
        FP_shadow = np.sum(zeros3 * (mask/255))
        FN_shadow = np.sum(zeros4 * (mask/255))
        ber = 1 - 0.5*(TP/(TP + FN) + TN/(TN + FP))
        # ber_shadow = 1 - 0.5*(TP_shadow/(TP_shadow + FN_shadow) + 1)
        ber_shadow = 1 - (TP_shadow/(TP_shadow + FN_shadow))
        ber_finals.append(ber)
        ber_final_shadows.append(ber_shadow)
        # print('{}_step_ber'.format(i), ber)
        # print('{}_step_ber_shadow'.format(i), ber_shadow)
        if ber < ber_final:
            ber_final = ber
        if ber_shadow < ber_final_shadow:
            ber_final_shadow = ber_shadow

    return acc_final, iou_final, ber_final, ber_final_shadow

    return np.mean(np.array(acc_finals)), np.mean(np.array(iou_finals)), np.mean(np.array(ber_finals))*100, np.mean(np.array(ber_final_shadows))*100














def acc(prediction_mask, gt_mask, step_num):
    predicted_mask = prediction_mask.copy()
    mask = gt_mask
    # print('acc mask', predicted_mask)
    step = 256 / step_num
    acc_final = 0
    acc_mask = np.zeros(np.shape(predicted_mask))
    for i in range(step_num):
        predicted_mask = prediction_mask.copy()
        predicted_mask[predicted_mask >= (i + 1) * step - 1] = 255
        predicted_mask[predicted_mask != 255] = 0
        acc_mask[predicted_mask==mask] = 1
        acc_mask[predicted_mask!=mask] = 0
        acc = np.sum(acc_mask) / int(np.shape(predicted_mask)[0]*np.shape(predicted_mask)[1]) * 100
        if acc > acc_final:
            acc_final = acc
    return acc_final



def iou(prediction_mask, gt_mask, step_num):
    predicted_mask = prediction_mask.copy()
    mask = gt_mask
    # print('iou mask', predicted_mask)
    step = 256 / step_num
    iou_final = 0
    for i in range(step_num):
        predicted_mask = prediction_mask.copy()
        predicted_mask[predicted_mask >= (i + 1) * step - 1] = 255
        predicted_mask[predicted_mask != 255] = 0
        total = predicted_mask/255 + mask/255
        zeros3 = np.zeros(np.shape(predicted_mask))
        zeros3[total == 2] = 1
        inter = np.sum(zeros3)
        zeros1 = np.zeros(np.shape(predicted_mask))
        zeros1[predicted_mask == 255] = 1
        predict_mask_area = np.sum(zeros1)
        zeros2 = np.zeros(np.shape(predicted_mask))
        zeros2[mask == 255] = 1
        gt_mask_area = np.sum(zeros2)
        mask_iou_value = inter / (predict_mask_area + gt_mask_area - inter)
        if mask_iou_value > iou_final:
            iou_final = mask_iou_value
    return iou_final


def OTSU_enhance(img_gray, th_begin=0, th_end=256, th_step=1):
    max_g = 0
    suitable_th = 0
    zeros1 = np.zeros(np.shape(img_gray))
    zeros1[img_gray>0] = 1
    img_size = np.sum(zeros1)

    for threshold in range(th_begin, th_end, th_step):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img*zeros1)
        back_pix = np.sum(bin_img_inv*zeros1)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue

        w0 = float(fore_pix) / img_size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold
    return suitable_th


def OTSU(img_array):            #传入的参数为ndarray形式
    height = img_array.shape[0]
    width = img_array.shape[1]
    count_pixel = np.zeros(256)

    for i in range(height):
        for j in range(width):
            count_pixel[int(img_array[i][j])] += 1
    max_variance = 0.0
    best_thresold = 0

    if np.max(img_array)!=0:
        min_value = np.min(img_array[img_array>0])
        max_value = np.max(img_array[img_array>0])
    else:
        min_value = np.min(img_array)
        max_value = np.max(img_array)


    for thresold in range(int(min_value), int(max_value)):
        # for thresold in range(0, 256):
        n0 = count_pixel[:thresold].sum()
        n1 = count_pixel[thresold:].sum()
        w0 = n0 / (height * width)
        w1 = n1 / (height * width)
        u0 = 0.0
        u1 = 0.0
        # for i in range(thresold):
        for i in range(int(min_value), thresold):
            u0 += i * count_pixel[i]
        # for j in range(thresold, 256):
        for j in range(thresold, int(max_value)):
            u1 += j * count_pixel[j]
        u = u0 * w0 + u1 * w1
        tmp_var = w0 * np.power((u - u0), 2) + w1 * np.power((u - u1), 2)
        if tmp_var > max_variance:
            best_thresold = thresold
            max_variance = tmp_var
    return best_thresold



# def otsu(gray):
#     pixel_number = gray.shape[0] * gray.shape[1]
#     mean_weigth = 1.0/pixel_number
#     # 发现bins必须写到257，否则255这个值只能分到[254,255)区间
#     his, bins = np.histogram(gray, np.arange(0,257))
#     final_thresh = -1
#     final_value = -1
#     intensity_arr = np.arange(256)
#     for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
#         pcb = np.sum(his[:t])
#         pcf = np.sum(his[t:])
#         Wb = pcb * mean_weigth
#         Wf = pcf * mean_weigth
#         mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
#         muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
#         value = Wb * Wf * (mub - muf) ** 2
#         if value > final_value:
#             final_thresh = t
#             final_value = value
#     return final_thresh

def ber_sklearn(prediction_mask, gt_mask, step_num):
    predicted_mask = prediction_mask.copy()
    mask = gt_mask

    ber_final = 1
    ber_final_shadow = 1
    ber_finals = []
    ber_final_shadows = []

    max_value = 255
    min_value = 0

    # max_value = np.max(predicted_mask)
    # min_value = np.min(predicted_mask)


    # print('max min value', max_value, min_value)
    step = (max_value - min_value) / step_num

    for i in range(step_num):
        predicted_mask = prediction_mask.copy()
        current_pixel_value = min_value + step*(i+1)
        predicted_mask[predicted_mask > current_pixel_value] = 255
        predicted_mask[predicted_mask != 255] = 0

        predicted_mask = np.reshape(predicted_mask/255, -1)
        mask = np.reshape(mask/255, -1)


        ber = 1 - balanced_accuracy_score(mask, predicted_mask)
        ber_shadow = 1 - 1 - balanced_accuracy_score(mask, predicted_mask)
        # print(ber, ber_shadow)
        ber_finals.append(ber)
        ber_final_shadows.append(ber_shadow)
        # print('{}_step_ber'.format(i), ber)
        # print('{}_step_ber_shadow'.format(i), ber_shadow)
        if ber < ber_final:
            ber_final = ber
        if ber_shadow < ber_final_shadow:
            ber_final_shadow = ber_shadow

    # print('maximum value', ber_final, ber_final_shadows)
    # print('mean value', np.mean(np.array(ber_finals))*100, np.mean(np.array(ber_final_shadows))*100)
    # return ber_final*100, ber_final_shadow*100
    # return np.mean(np.array(ber_finals))*100, np.mean(np.array(ber_final_shadows))*100
    return ber_finals, ber_final_shadows








def ber_ratioshreshold(prediction_mask, gt_mask, step_num):
    predicted_mask = prediction_mask.copy()
    mask = gt_mask

    ber_final = 1
    ber_final_shadow = 1
    ber_finals = []
    ber_final_shadows = []


    # print('max min value', max_value, min_value)
    # step = (max_value - min_value) / step_num

    predict_pixels = (np.sort(np.reshape(predicted_mask,-1)))
    # print(np.shape(predict_pixels))
    threshold_g_gt = np.mean(mask)
    mask[mask>=threshold_g_gt] = 255
    mask[mask!=255] = 0
    ratio = np.sum(mask/255)
    threshold_g_pre = predict_pixels[int(-ratio)]+1e-5
    # print('threshold', threshold_g_pre)

    predicted_mask[predicted_mask >= threshold_g_pre]  = 255
    predicted_mask[predicted_mask!=255] = 0

    total = predicted_mask/255 + mask/255
    zeros1 = np.zeros(np.shape(predicted_mask))
    zeros2 = np.zeros(np.shape(predicted_mask))
    zeros3 = np.zeros(np.shape(predicted_mask))
    zeros4 = np.zeros(np.shape(predicted_mask))
    ####[1,1]
    zeros1[total == 2] = 1
    ####[0,0]
    zeros2[total == 0] = 1
    ####[1,0]
    zeros3[(predicted_mask/255 - mask/255) == 1] = 1
    ####[0,1]
    zeros4[(predicted_mask/255 - mask/255) == -1] = 1

    TP = np.sum(zeros1)
    TN = np.sum(zeros2)
    FP = np.sum(zeros3)
    FN = np.sum(zeros4)
    # print('gggg', TP, TN, FP, FN, TP+TN+FP+FN, TP + FN, TN + FP)
    TP_shadow = np.sum(zeros1 * (mask/255))
    TN_shadow = np.sum(zeros2 * (mask/255))
    FP_shadow = np.sum(zeros3 * (mask/255))
    FN_shadow = np.sum(zeros4 * (mask/255))

    ber = 1 - 0.5*(TP/(TP + FN) + TN/(TN + FP))
    # ber_shadow = 1 - 0.5*(TP_shadow/(TP_shadow + FN_shadow) + 1)
    ber_shadow = 1 - (TP_shadow/(TP_shadow + FN_shadow))
    # print(ber, ber_shadow)
    ber_finals.append(ber)
    ber_final_shadows.append(ber_shadow)
    return ber_finals, ber_final_shadows


def ber_ratioshreshold1(prediction_mask, gt_mask, step_num):


    ber_final = 1
    ber_final_shadow = 1
    ber_finals = []
    ber_final_shadows = []

    predicted_mask = prediction_mask.copy()
    mask = gt_mask.copy()

    ######mask bbox
    # coors = np.where(mask == 255)
    # index_row_min = np.min(coors[0])
    # index_row_max = np.max(coors[0])
    # index_col_min = np.min(coors[1])
    # index_col_max = np.max(coors[1])
    # valid_area = mask.copy()/255
    # valid_area[index_row_min:index_row_max, index_col_min:index_col_max,0] = 1

    ####global
    # predict_pixels = np.unique(np.sort(predicted_mask))
    # threshold_g_gt = np.mean(mask)
    # mask[mask>=threshold_g_gt] = 255
    # mask[mask!=255] = 0
    # ratio = np.sum(mask/255) / (256*256)
    # threshold_g_pre = predict_pixels[int(len(predict_pixels) * (1 - ratio))]
    # threshold_g_pre = np.mean(predicted_mask)
    # print(np.shape(predict_pixels))
    predict_pixels = (np.sort(np.reshape(predicted_mask,-1)))
    # print(np.shape(predict_pixels))
    threshold_g_gt = np.mean(mask)
    mask[mask>=threshold_g_gt] = 255
    mask[mask!=255] = 0
    ratio = np.sum(mask/255)
    threshold_g_pre = predict_pixels[int(-ratio)]+1e-5

    predicted_mask[predicted_mask >= threshold_g_pre]  = 255
    predicted_mask[predicted_mask!=255] = 0
    # print('threshold, ratio', threshold_g_pre, ratio)
    total = predicted_mask/255 + mask/255
    zeros1 = np.zeros(np.shape(predicted_mask))
    zeros2 = np.zeros(np.shape(predicted_mask))
    zeros3 = np.zeros(np.shape(predicted_mask))
    zeros4 = np.zeros(np.shape(predicted_mask))
    total_area = np.zeros(np.shape(predicted_mask))
    total_area[total>0] = 1
    ####[1,1]
    zeros1[total == 2] = 1
    ####[0,0]
    zeros2[total == 0] = 1
    ####[1,0]
    zeros3[(predicted_mask/255 - mask/255) == 1] = 1
    ####[0,1]
    zeros4[(predicted_mask/255 - mask/255) == -1] = 1
    TP = np.sum(zeros1)
    TN = np.sum(zeros2)
    FP = np.sum(zeros3)
    FN = np.sum(zeros4)
    ber = 1 - 0.5*(TP/(TP + FN) + TN/(TN + FP))


    # print('ber', ber)
    ber_finals.append(ber)
    # ber_final_shadows.append(ber_shadow)
    # print('{}_step_ber'.format(i), ber)
    # print('{}_step_ber_shadow'.format(i), ber_shadow)
    if ber < ber_final:
        ber_final = ber
    # if ber_shadow < ber_final_shadow:
    #     ber_final_shadow = ber_shadow

    # print('maximum value', ber_final, ber_final_shadows)
    # print('mean value', np.mean(np.array(ber_finals))*100, np.mean(np.array(ber_final_shadows))*100)
    # return ber_final*100, ber_final_shadow*100
    # return np.mean(np.array(ber_finals))*100, np.mean(np.array(ber_final_shadows))*100
    return ber_finals



def ber(prediction_mask, gt_mask, step_num):
    predicted_mask = prediction_mask.copy()
    mask = gt_mask
    # print('ber mask', predicted_mask)
    step = 256 / step_num
    ber_final = 1
    ber_final_shadow = 1
    ber_finals = []
    ber_final_shadows = []

    for i in range(step_num):
        predicted_mask = prediction_mask.copy()
        predicted_mask[predicted_mask > (i + 1) * step - 1] = 255
        predicted_mask[predicted_mask != 255] = 0
        total = predicted_mask/255 + mask/255
        zeros1 = np.zeros(np.shape(predicted_mask))
        zeros2 = np.zeros(np.shape(predicted_mask))
        zeros3 = np.zeros(np.shape(predicted_mask))
        zeros4 = np.zeros(np.shape(predicted_mask))
        ####[1,1]
        zeros1[total == 2] = 1
        ####[0,0]
        zeros2[total == 0] = 1
        ####[1,0]
        zeros3[(predicted_mask/255 - mask/255) == 1] = 1
        ####[0,1]
        zeros4[(predicted_mask/255 - mask/255) == -1] = 1

        TP = np.sum(zeros1)
        TN = np.sum(zeros2)
        FP = np.sum(zeros3)
        FN = np.sum(zeros4)
        # print('gggg', TP, TN, FP, FN, TP+TN+FP+FN, TP + FN, TN + FP)
        TP_shadow = np.sum(zeros1 * (mask/255))
        TN_shadow = np.sum(zeros2 * (mask/255))
        FP_shadow = np.sum(zeros3 * (mask/255))
        FN_shadow = np.sum(zeros4 * (mask/255))

        ber = 1 - 0.5*(TP/(TP + FN) + TN/(TN + FP))
        # ber_shadow = 1 - 0.5*(TP_shadow/(TP_shadow + FN_shadow) + 1)
        ber_shadow = 1 - (TP_shadow/(TP_shadow + FN_shadow))
        ber_finals.append(ber)
        ber_final_shadows.append(ber_shadow)
        # print('{}_step_ber'.format(i), ber)
        # print('{}_step_ber_shadow'.format(i), ber_shadow)
        if ber < ber_final:
            ber_final = ber
        if ber_shadow < ber_final_shadow:
            ber_final_shadow = ber_shadow

    # print('maximum value', ber_final, ber_final_shadows)
    # print('mean value', np.mean(np.array(ber_finals))*100, np.mean(np.array(ber_final_shadows))*100)
    return ber_final*100, ber_final_shadow*100
    # return np.mean(np.array(ber_finals))*100, np.mean(np.array(ber_final_shadows))*100




def compute_total_variation_loss(img, weight):
    tv_h = ((img[:,:,1:,:] - img[:,:,:-1,:]).pow(2)).sum()
    tv_w = ((img[:,:,:,1:] - img[:,:,:,:-1]).pow(2)).sum()
    return weight * (tv_h + tv_w)


def light_theta(x1, y1, x2, y2):
    theta = math.atan((y2 - y1) / (x2 - x1 + 1e-6))
    return theta


def Light_direction_calculator(instance_mask, shadow_mask, gt_light_direction, device):
    # instance_mask = util.tensor2cvim(instance_mask)
    # shadow_mask = util.tensor2cvim(shadow_mask)
    # print('instance',np.shape(instance_mask))
    # print('shadow',np.shape(shadow_mask))
    # instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_BGR2GRAY)
    # shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
    _, instance_mask = cv2.threshold(instance_mask, 0, 255, 0)
    _, shadow_mask = cv2.threshold(shadow_mask, 0, 255, 0)
    contours_instance_mask, _ = cv2.findContours(instance_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_shadow_mask, _ = cv2.findContours(shadow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_instance_mask) != 1 or len(contours_shadow_mask) != 1:
        # print('failed')
        bbox_theta = gt_light_direction
    else:

        # print(np.shape(instance_mask))
        bbox_shadowmask = cv2.boundingRect(contours_shadow_mask[0])
        bbox_instancemask = cv2.boundingRect(contours_instance_mask[0])
        bbox_theta = light_theta(bbox_instancemask[0] + bbox_instancemask[2] / 2,
                                 bbox_instancemask[1] + bbox_instancemask[3] / 2, \
                                 bbox_shadowmask[0] + bbox_shadowmask[2] / 2,
                                 bbox_shadowmask[1] + bbox_shadowmask[3] / 2)
        # return torch.from_numpy(np.ndarray(bbox_theta))
        # return torch.float(bbox_theta)
        bbox_theta = torch.FloatTensor(np.array(bbox_theta))
    return bbox_theta.to(device)


def Light_bbox_theta(object_mask, shadow_mask):
    # bbox [x,y,w,h]
    _, object_mask = cv2.threshold(object_mask, 0, 255, 0)
    bbox_object = cv2.boundingRect(object_mask)
    object_center_x = bbox_object[0] + bbox_object[2] / 2
    object_center_y = bbox_object[1] + bbox_object[3] / 2
    _, shadow_mask = cv2.threshold(shadow_mask, 0, 255, 0)
    bbox_shadowmask = cv2.boundingRect(shadow_mask)
    shadow_x = bbox_shadowmask[0] + bbox_shadowmask[2] / 2
    shadow_y = bbox_shadowmask[1] + bbox_shadowmask[3] / 2
    bbox_theta_pred = light_theta(object_center_x, object_center_y, shadow_x, shadow_y)
    bbox_theta_pred = torch.FloatTensor(np.array(bbox_theta_pred))
    return bbox_theta_pred


def Bbox_regression(shadow_mask):
    # bbox [x,y,w,h]
    _, shadow_mask = cv2.threshold(shadow_mask, 0, 255, 0)
    bbox_shadowmask = cv2.boundingRect(shadow_mask)
    # shadow_x = bbox_shadowmask[0] + bbox_shadowmask[2]/2
    # shadow_y = bbox_shadowmask[1] + bbox_shadowmask[3]/2
    # bbox_theta_pred = light_theta(object_center_x, object_center_y, shadow_x,shadow_y)
    bbox_shadowmask = torch.IntTensor(np.array(bbox_shadowmask))
    bbox_shadowmask[0] = bbox_shadowmask[0]
    bbox_shadowmask[1] = bbox_shadowmask[0] + bbox_shadowmask[1]
    bbox_shadowmask[2] = bbox_shadowmask[2]
    bbox_shadowmask[3] = bbox_shadowmask[2] + bbox_shadowmask[3]
    # print(bbox_shadowmask.size())
    return bbox_shadowmask


class SIDModel(DistangleModel):
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

        self.loss_names = ['rescontruction']
        if self.opt.illumination:
            self.loss_names.append('G_param')
            self.loss_names.append('G_MSP')
            self.loss_names.append('rescontruction_gt')
        if self.opt.bg_maskattention:
            self.loss_names.append('bg_instance_attention')
            self.loss_names.append('bg_shadow_attention')
        if self.opt.lambda_GAN:
            self.loss_names.append('D')
            self.loss_names.append('G_GAN')
        if self.opt.lambda_GAN_mask:
            self.loss_names.append('D_mask')
            self.loss_names.append('G_GAN_mask')
        if self.opt.lambda_REF1:
            self.loss_names.append('refine')
        if self.opt.lambda_TV1:
            self.loss_names.append('lv')


        self.visual_names = ['shadowfree_img', 'shadow_img','instance_mask', 'shadow_mask', 'redark', 'shadowmask_pred',  'alpha_pred_vis', 'diff_alpha_mask', 'final']



        # self.visual_names = ['shadowfree_img', 'shadow_img', 'instance_mask', 'shadow_mask', 'bg_instance_mask' , 'bg_shadow_mask' , 'shadowmask_pred', 'shadow_mask_predict','shadow_diff', 'edge','final']
        # if self.opt.illumination:
        #     if self.opt.softattention:
        #         self.visual_names.append('shadowmask_pred_coarse')
        #     # self.visual_names.append('shadowmask_pred')
        #     self.visual_names.append('alpha_pred')
        # if self.opt.bg_maskattention:
        #     self.visual_names.append('bg_instance_attention')
        #     self.visual_names.append(' bg_shadow_attention')
        # if self.opt.bg_selfattention:
        #     self.visual_names.append('bg_pure_mask')
        if self.opt.lambda_REF1:
            self.visual_names.append('residual_final')


        # self.visual_names = ['shadowfree_img', 'shadowfree_img_pure',  'shadow_img', 'instance_mask', 'shadow_mask', 'bg_instance_mask', 'bg_shadow_mask'  , 'final']


        # self.visual_names = ['shadowfree_img', 'shadow_img', 'instance_mask', 'shadow_mask', 'final']

        # self.visual_names = ['final']


        self.model_names = ['MSP']
        if self.opt.illumination:
            if self.opt.softattention:
                self.model_names.append('SA')
            self.model_names.append('G')
            self.model_names.append('M')
        if self.opt.bg_maskattention:
            self.model_names.append('ATT')
        if self.opt.lambda_GAN:
            self.model_names.append('D')
        if self.opt.lambda_GAN_mask:
            self.model_names.append('D_mask')
        if self.opt.lambda_REF1:
            self.model_names.append('REF')
        # load/define networks

        opt.output_nc = 3
        if self.opt.lambda_REF1:
            self.netREF = networks.define_G(3, 3, opt.ngf, 'unet_32', opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        ####softattention on predicted shadow mask
        if self.opt.softattention and self.opt.illumination:
            self.netSA = networks.define_G(4, 1, opt.ngf, 'SA', opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.opt.bg_maskattention:
            self.netATT = networks.define_G(4, 1, opt.ngf, 'Res_twobranch1', opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

            if self.opt.illumination:
                self.netMSP = networks.define_G(6, 1, opt.ngf, 'unet_32', opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netG = networks.define_G(6, 6, opt.ngf, 'RESNEXT18', opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

                self.netM = networks.define_G(7, 3, opt.ngf, 'unet_32', opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netMSP = networks.define_G(6, 3, opt.ngf, 'unet_32', opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)



        elif self.opt.bg_selfattention:
            if self.opt.illumination:
                ######shadow mask predictor selection: non-local attention integration layer
                #[m_fo, m_bgpure, m_bos]
                self.netMSP = networks.define_G(4, 1, opt.ngf, 'unet_attention', opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                # self.netMSP = networks.define_G(4, 1, opt.ngf, 'Attention_Twobranch_UNet', opt.norm,
                #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                # self.netMSP = networks.define_G(4, 1, opt.ngf, 'Attention_Twobranch_UNet_Noshare', opt.norm,
                #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                ######shadow parameter predictor
                self.netG = networks.define_G(6, 6, opt.ngf, 'RESNEXT18', opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                # self.netG = networks.define_G(6, 6, opt.ngf, 'simpleCNN', opt.norm,
                #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

                self.netM = networks.define_G(7, 3, opt.ngf, 'unet_32', opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netMSP = networks.define_G(4, 3, opt.ngf, 'unet_attention', opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        else:
            if self.opt.illumination:
                self.netMSP = networks.define_G(4, 1, opt.ngf, 'unet_32', opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netG = networks.define_G(4, 6, opt.ngf, 'RESNEXT18', opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

                self.netM = networks.define_G(7, 3, opt.ngf, 'unet_32', opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netMSP = networks.define_G(4, 3, opt.ngf, 'unet_32', opt.norm,
                                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)




        if self.opt.lambda_GAN:
            if self.opt.conditionD:
                if self.opt.illumination:
                    self.netD = networks.define_D(5, opt.ndf, opt.netD, 5, opt.norm, False, opt.init_type, opt.init_gain,
                                                  self.gpu_ids)
                else:
                    self.netD = networks.define_D(4, opt.ndf, opt.netD, 5, opt.norm, False, opt.init_type,
                                                  opt.init_gain, self.gpu_ids)
            else:
                self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, 5, opt.norm, False, opt.init_type,
                                              opt.init_gain, self.gpu_ids)

        if self.opt.lambda_GAN_mask:
            self.netD_mask = networks.define_D(5, opt.ndf, opt.netD, 5, opt.norm, False, opt.init_type, opt.init_gain,
                                               self.gpu_ids)

        self.netMSP.to(self.device)
        if self.opt.illumination:
            if self.opt.softattention:
                self.netSA.to(self.device)
            self.netG.to(self.device)
            self.netM.to(self.device)
        if self.opt.bg_maskattention:
            self.netATT.to(self.device)
        if self.opt.lambda_GAN:
            self.netD.to(self.device)
        if self.opt.lambda_GAN_mask:
            self.netD_mask.to(self.device)
        if self.opt.lambda_REF1:
            self.netREF.to(self.device)


        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.no_lsgan).to(self.device)
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_MSP = torch.optim.Adam(self.netMSP.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            if self.opt.illumination:
                if self.opt.softattention:
                    self.optimizer_SA = torch.optim.Adam(self.netSA.parameters(),
                                                         lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)

                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
                self.optimizer_M = torch.optim.Adam(self.netM.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            if self.opt.bg_maskattention:
                self.optimizer_ATT = torch.optim.Adam(self.netATT.parameters(),
                                                      lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)

            if self.opt.lambda_GAN:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            if self.opt.lambda_GAN_mask:
                self.optimizer_D_mask = torch.optim.Adam(self.netD_mask.parameters(),
                                                         lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=1e-5)

            if self.opt.lambda_REF1:
                self.optimizer_REF = torch.optim.Adam(self.netREF.parameters(),
                                                      lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)



            self.optimizers.append(self.optimizer_MSP)
            if self.opt.illumination:
                if self.opt.softattention:
                    self.optimizers.append(self.optimizer_SA)
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_M)
            if self.opt.bg_maskattention:
                self.optimizers.append(self.optimizer_ATT)
            if self.opt.lambda_GAN:
                self.optimizers.append(self.optimizer_D)
            if self.opt.lambda_GAN_mask:
                self.optimizers.append(self.optimizer_D_mask)
            if self.opt.lambda_REF1:
                self.optimizers.append(self.optimizer_REF)



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
        self.light_direction = input['light'].to(self.device).type(torch.float)
        self.shadow_param = input['param'].to(self.device).type(torch.float)
        self.light_image = input['light_image'].to(self.device)
        self.bg_instance_mask = input['bg_instance'].to(self.device)
        self.bg_shadow_mask = input['bg_shadow'].to(self.device)

        # self.shadowfree_img_pure = input['CC'].to(self.device)

        self.bg_mask = ((self.bg_instance_mask/2+0.5) + (self.bg_shadow_mask/2+0.5))*2 - 1
        self.bg_pure_mask = (1 - (self.bg_mask/2+0.5) - (self.instance_mask/2+0.5) )*2 - 1


        ####abalation study for background branch
        # self.bg_mask = torch.zeros(self.instance_mask.size())
        # self.bg_instance_mask = torch.zeros(self.instance_mask.size()).to(self.device)
        # self.bg_shadow_mask = torch.zeros(self.instance_mask.size()).to(self.device)
        # self.bg_pure_mask = torch.zeros(self.instance_mask.size()).to(self.device)
        self.edge = input['edge'].to(self.device)






    def forward(self):
        if self.opt.residual:
            self.final_residual = self.netMSP(torch.cat([self.shadowfree_img, self.instance_mask],1))
            self.final = self.final_residual + self.shadowfree_img
            if self.opt.lambda_REF1:
                self.final_residual_refine = self.netREF(self.final_residual)
                self.final_refine = self.final_residual_refine + self.shadowfree_img
        else:
            # self.final = self.netMSP(torch.cat([self.shadowfree_img, self.instance_mask],1))
            ####shadowgan baselines
            self.multiply = self.netMSP(torch.cat([self.shadowfree_img, self.instance_mask],1))
            self.final = self.shadowfree_img * self.multiply
            if self.opt.lambda_REF1:
                self.final_refine = self.netREF(self.final)


        if self.opt.residual:
            self.final_residual = (self.shadowfree_img / 2 + 0.5) * (1 - self.alpha_pred) + self.dark * (self.alpha_pred)
            self.final_residual = self.final_residual * 2 - 1
            self.final = self.final_residual + self.shadowfree_img
            if self.opt.lambda_REF1:
                self.final_residual_refine = self.netREF(self.final_residual)
                self.final_refine = self.final_residual_refine + self.shadowfree_img
        else:
            self.final = (self.shadowfree_img / 2 + 0.5) * (1 - self.alpha_pred) + self.dark * (self.alpha_pred)
            self.final = self.final * 2 - 1

            self.final_gt = (self.shadowfree_img / 2 + 0.5) * (1 - self.alpha_pred_gt) + self.dark * (self.alpha_pred_gt)
            self.final_gt = self.final_gt * 2 - 1
            
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


        if self.opt.illumination:
            self.shadowmask_pred = self.shadowmask_pred
            if self.opt.softattention:
                self.shadowmask_pred = self.shadowmask_pred_coarse

        else:
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

    def backward_D_mask(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_mask = self.shadowmask_pred  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake_mask = self.netD_mask(torch.cat([self.shadowfree_img, fake_mask.detach(), self.instance_mask],1) )
        self.loss_D_fake_mask = self.criterionGAN(pred_fake_mask, False)
        # Real
        real_mask = self.shadow_mask
        pred_real_mask = self.netD_mask(torch.cat([self.shadowfree_img, real_mask, self.instance_mask],1 ))
        self.loss_D_real_mask = self.criterionGAN(pred_real_mask, True)
        # combine loss and calculate gradients
        self.loss_D_mask = (self.loss_D_fake_mask + self.loss_D_real_mask) * 0.5
        self.loss_D_mask.backward()


    def backward(self):
        #####GAN loss
        self.loss_rescontruction = self.MSELoss(self.final, self.shadow_img)
        self.loss = self.loss_rescontruction * self.opt.lambda_I1

        if self.opt.lambda_GAN:
            pred_real = self.netD(self.final)
            self.loss_G_GAN = self.criterionGAN(pred_real, True)
            self.loss += self.loss_G_GAN * self.opt.lambda_GAN

        
        self.loss_G_MSP = self.MSELoss(self.shadowmask_pred * (self.edge / 2 + 0.5),
                                       self.shadow_mask * (self.edge / 2 + 0.5)) * 10
        self.loss_G_MSP += self.MSELoss(self.shadowmask_pred * (1 - (self.edge / 2 + 0.5)),
                                        self.shadow_mask * (1 - (self.edge / 2 + 0.5)))


        self.loss_rescontruction_gt = self.MSELoss(self.final_gt, self.shadow_img)

        self.loss += self.loss_rescontruction_gt * self.opt.lambda_I1
        self.loss += self.loss_G_param * self.opt.lambda_P1
        self.loss += self.loss_G_MSP * self.opt.lambda_M1

        self.loss.backward()

    def optimize_parameters(self):
        if self.isTrain:
            self.forward()
            if self.opt.lambda_GAN:
                #####update discriminator
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.optimizer_D.zero_grad()  # set D's gradients to zero
                self.backward_D()  # calculate gradients for D
                self.optimizer_D.step()  # update D's weights
            if self.opt.lambda_GAN_mask:
                #####update discriminator
                self.set_requires_grad(self.netD_mask, True)  # enable backprop for D
                self.optimizer_D_mask.zero_grad()  # set D's gradients to zero
                self.backward_D_mask()  # calculate gradients for D
                self.optimizer_D_mask.step()  # update D's weights


            ######update generator
            if self.opt.lambda_GAN:
                self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            if self.opt.lambda_GAN_mask:
                self.set_requires_grad(self.netD_mask, False)  # D_mask requires no gradients when optimizing G

            self.optimizer_MSP.zero_grad()
            self.backward()
            self.optimizer_MSP.step()
        else:
            self.forward()

    def get_light_direction(self):
        nim = self.shadowfree_img.shape[0]
        light_preds = torch.zeros(nim).to(self.device)
        for i in range(nim):
            current_instance_mask = util.tensor2cvim(self.instance_mask.data[i:i + 1, :, :, :])
            current_shadow_mask = util.tensor2cvim(self.shadowmask_pred.data[i:i + 1, :, :, :])
            current_light = Light_direction_calculator(current_instance_mask, current_shadow_mask,
                                                       self.light_direction.data[i:i + 1], self.device)
            light_preds[i] = current_light
        return light_preds

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

        RMSE_new = []
        shadowRMSE_new = []


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
            predicted_mask = util.tensor2imonechannel(getattr(self, 'shadow_mask_predict').data[i:i + 1, :, :, :])

            ####RMSE
            RMSE.append(np.mean((gt - prediction)**2))
            shadowRMSE.append(np.sum(((gt - prediction)**2)*(mask/255)) / (np.sum(mask/255) * 3))

            ###new
            RMSE_new.append(math.sqrt(compare_mse(gt, prediction)))
            shadowRMSE_new.append(math.sqrt(compare_mse(gt*(mask/255), prediction*(mask/255))*256*256/np.sum(mask/255)))
            # RMSE.append(math.sqrt(np.mean((gt - prediction)**2)))

            ####torch version
            gt_tensor = (getattr(self, 'shadow_img').data[i:i + 1, :, :, :]/2 + 0.5) * 255
            prediction_tensor = (getattr(self, 'final').data[i:i + 1, :, :, :]/2 + 0.5) * 255
            mask_tensor = (getattr(self, 'shadow_mask').data[i:i + 1, :, :, :]/2 + 0.5)
            SSIM.append(pytorch_ssim.ssim(gt_tensor, prediction_tensor, window_size = 11, size_average = True))
            shadowSSIM.append(ssim.ssim(gt_tensor, prediction_tensor,mask=mask_tensor))

        # return RMSE,shadowRMSE, SSIM, shadowSSIM

        return RMSE,shadowRMSE, SSIM, shadowSSIM,RMSE_new,shadowRMSE_new




    def get_prediction(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        inputG = torch.cat([self.input_img, self.shadow_mask], 1)
        self.shadow_mask = (self.shadow_mask > 0.9).type(torch.float) * 2 - 1
        self.shadow_mask_3d = (self.shadow_mask > 0).type(torch.float).expand(self.input_img.shape)

        inputG = F.upsample(inputG, size=(256, 256))
        self.shadow_param_pred = self.netG(inputG)
        w = self.input_img.shape[2]
        h = self.input_img.shape[3]
        n = self.input_img.shape[0]
        m = self.input_img.shape[1]
        self.shadow_param_pred = self.shadow_param_pred.view([n, 6, -1])
        self.shadow_param_pred = torch.mean(self.shadow_param_pred, dim=2)
        self.shadow_param_pred[:, [1, 3, 5]] = (self.shadow_param_pred[:, [1, 3, 5]] * 2) + 3

        self.lit = self.input_img.clone() / 2 + 0.5
        add = self.shadow_param_pred[:, [0, 2, 4]]
        mul = self.shadow_param_pred[:, [1, 3, 5]]
        # mul = (mul +2) * 5/3
        n = self.shadow_param_pred.shape[0]
        add = add.view(n, 3, 1, 1).expand((n, 3, w, h))
        mul = mul.view(n, 3, 1, 1).expand((n, 3, w, h))
        self.lit = self.lit * mul + add
        self.out = (self.input_img / 2 + 0.5) * (1 - self.shadow_mask_3d) + self.lit * self.shadow_mask_3d
        self.out = self.out * 2 - 1

        inputM = torch.cat([self.input_img, self.lit, self.shadow_mask], 1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred + 1) / 2
        # self.alpha_pred_3d=  self.alpha_pred.repeat(1,3,1,1)

        self.final = (self.input_img / 2 + 0.5) * (1 - self.alpha_pred) + self.lit * self.alpha_pred
        self.final = self.final * 2 - 1

        RES = dict()
        RES['final'] = util.tensor2im(self.final, scale=0)
        return RES

