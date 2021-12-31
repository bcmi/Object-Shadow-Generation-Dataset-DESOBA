import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image,ImageChops
from PIL import ImageFilter
import torch
from pdb import set_trace as st
import random
import numpy as np
import cv2
import time
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import itertools


######'dir_A': shadowimage
######'dir_B': shadowmask
######'dir_C': shadowfree
######'dir_param':illumination parameter
######'dir_light': light direction
######'dir_instance':object mask

def resize_pos(bbox, src_size,tar_size):
    x1,y1,x2,y2 = bbox
    w1=src_size[0]
    h1=src_size[1]
    w2=tar_size[0]
    h2=tar_size[1]
    y11= int((h2/h1)*y1)
    x11=int((w2/w1)*x1)
    y22=int((h2/h1)*y2)
    x22=int((w2/w1)*x2)
    return [x11, y11, x22, y22]

def mask_to_bbox(mask, specific_pixels, new_w, new_h):
    #[w,h,c]
    w,h = np.shape(mask)[:2]
    valid_index = np.argwhere(mask==specific_pixels)[:,:2]
    if np.shape(valid_index)[0] < 1:
        x_left = 0
        x_right = 0
        y_bottom = 0
        y_top = 0
    else:
        x_left = np.min(valid_index[:,0])
        x_right = np.max(valid_index[:,0])
        y_bottom = np.min(valid_index[:,1])
        y_top = np.max(valid_index[:,1])
    origin_box = [x_left, y_bottom, x_right, y_top]
    resized_box = resize_pos(origin_box, [w,h], [new_w, new_h])
    return resized_box

def bbox_to_mask(box,mask_plain):
    mask_plain[box[0]:box[2], box[1]:box[3]] = 255
    return mask_plain


class ShadowGenerationDatasetInference2dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.is_train = self.opt.isTrain
        self.root = opt.dataroot
        self.dir_shadowfree = opt.shadowfree_path #os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_fg_instance = opt.instance_path
        
        
        self.birdy_deshadoweds = []
        self.birdy_fg_instances = []
        



        # for root,_,fnames in sorted(os.walk(self.dir_shadowimg)):
        for root,_,fnames in sorted(os.walk(self.dir_shadowfree)):
            fname_int = [int(fname.split('.')[0]) for fname in fnames]
            for name in sorted(fname_int,key=int):
                fname = str(name) + '.png'
                if fname.endswith('.png'):
                    X=dict()
                    X['shadowfree_path'] = os.path.join(self.dir_shadowfree,fname)
                    X['fginstance_path'] = os.path.join(self.dir_fg_instance,fname)
                    

                    shadowfree = Image.open(X['shadowfree_path']).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
                    instance = Image.open(X['fginstance_path']).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
                    
                    self.birdy_deshadoweds.append(shadowfree)
                    self.birdy_fg_instances.append(instance)
                    
                           
        self.data_size = len(self.birdy_deshadoweds)
        print('datasize', self.data_size)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=opt.norm_mean,
                                               std = opt.norm_std)]

        self.transformA = transforms.Compose(transform_list)
        self.transformB = transforms.Compose([transforms.ToTensor()])

        self.transformAugmentation = transforms.Compose([
            transforms.Resize(int(self.opt.loadSize * 1.12), Image.BICUBIC),
            transforms.RandomCrop(self.opt.loadSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    def __getitem__(self,index):
        birdy = {}
        birdy['A'] = self.birdy_deshadoweds[index]
        birdy['C'] = self.birdy_deshadoweds[index]
        birdy['instancemask'] = self.birdy_fg_instances[index]

        zeros_mask = Image.fromarray(np.uint8(np.array(np.zeros(np.shape(np.array(birdy['instancemask']))))), mode='L')
        #setting background instance mask and background shadow mask as zeros
        birdy['B'] = zeros_mask
        #setting foreground shadow mask as zeros
        birdy['bg_shadow'] =  zeros_mask
        birdy['bg_instance'] = zeros_mask
       
        ow = birdy['A'].size[0]
        oh = birdy['A'].size[1]
        loadSize = self.opt.loadSize
        if self.opt.randomSize:
            loadSize = np.random.randint(loadSize + 1,loadSize * 1.3 ,1)[0]
        if self.opt.keep_ratio:
            if w>h:
                ratio = np.float(loadSize)/np.float(h)
                neww = np.int(w*ratio)
                newh = loadSize
            else:
                ratio = np.float(loadSize)/np.float(w)
                neww = loadSize
                newh = np.int(h*ratio)
        else:
            neww = loadSize
            newh = loadSize


        if not self.is_train:
            for k,im in birdy.items():
                birdy[k] = im.resize((neww, newh),Image.NEAREST)

        if self.opt.no_flip and self.opt.no_crop and self.opt.no_rotate:
            for k,im in birdy.items():
                birdy[k] = im.resize((neww, newh),Image.NEAREST)

        #### flip
        if not self.opt.no_flip:
            for i in ['A', 'B', 'C',  'instancemask', 'bg_shadow', 'bg_instance']:
                birdy[i] = birdy[i].transpose(Image.FLIP_LEFT_RIGHT)


        for k,im in birdy.items():
            birdy[k] = self.transformB(im)


        for i in ['A', 'B', 'C',  'instancemask', 'bg_shadow', 'bg_instance']:
            if i in birdy:
                birdy[i] = (birdy[i] - 0.5)*2


        h = birdy['A'].size()[1]
        w = birdy['A'].size()[2]
        if not self.opt.no_crop:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            for k, im in birdy.items():
                birdy[k] = im[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
                birdy[k] = im.type(torch.FloatTensor)


        for k,im in birdy.items():
            im = F.interpolate(im.unsqueeze(0), size = self.opt.loadSize)
            birdy[k] = im.squeeze(0)


        birdy['w'] = ow
        birdy['h'] = oh
        #setting param as zeros
        shadow_param=[0,0,0,0,0,0]
        birdy['param'] = torch.FloatTensor(np.array(shadow_param))
        

        #if the shadow area is too small, let's not change anything:
        return birdy
    def __len__(self):
        return self.data_size

    def name(self):
        return 'ShadowGenerationDatasetInference2'
