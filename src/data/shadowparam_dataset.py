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



def generate_training_pairs(newwh, shadow_image, deshadowed_image, instance_mask, shadow_mask, new_shadow_mask, shadow_param,imname_list, is_train, \
                            birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows, \
                            birdy_bg_instances,  birdy_bg_shadows, birdy_edges, birdy_shadowparas, birdy_shadow_object_ratio, birdy_instance_boxes, birdy_shadow_boxes, birdy_instance_box_areas, birdy_shadow_box_areas,birdy_im_lists):

    ####producing training/test pairs according pixel value
    instance_pixels_a = np.unique(np.sort(instance_mask[instance_mask>0]))
    shadow_pixels_a = np.unique(np.sort(shadow_mask[shadow_mask>0]))
    instance_pixels = np.intersect1d(instance_pixels_a,shadow_pixels_a)

    object_num = len(instance_pixels)
    if object_num==1:
        object_num=2


    if not is_train:
        object_num += 1

    for i in range(1, object_num):
        selected_instance_pixel_combine = itertools.combinations(instance_pixels, i)
        if not is_train:
            #####combination
            ###selecting one foreground image
            if i!=1:
                continue
            ####selecting two foreground image
            # if i!=2:
            #     continue

            # ####1,2 all includse
            # if i>2:
            #     continue

        else:
            ## using 1 or 2 objects as foreground objects
            if i > 3:
                continue
            
        for combine in selected_instance_pixel_combine:
            fg_instance = instance_mask.copy()
            fg_shadow = shadow_mask.copy()
            bg_instance = instance_mask.copy()
            bg_shadow = shadow_mask.copy()
            
            ###removing shadow without object for foreground object
            fg_shadow[fg_shadow==255] = 0
            fg_instance_boxes = []
            fg_shadow_boxes = []
            remaining_fg_pixel = list(set(instance_pixels).difference(set(combine)))
            # producing foreground object mask
            for pixel in combine:
                area = ( fg_shadow== pixel).sum()
                total_area = (fg_shadow > -1).sum()
                fg_shadow_boxes.append(mask_to_bbox(fg_shadow, pixel, newwh, newwh))
                fg_shadow[fg_shadow==pixel] = 255
                fg_instance_boxes.append(mask_to_bbox(fg_instance,pixel,newwh, newwh))
                fg_instance[fg_instance==pixel] = 255
            fg_shadow[fg_shadow!=255] = 0
            fg_instance[fg_instance!=255] = 0

            for pixel in remaining_fg_pixel:
                bg_instance[bg_instance==pixel]=255
                bg_shadow[bg_shadow==pixel]=255
            bg_instance[bg_instance!=255] = 0
            bg_shadow[bg_shadow!=255] = 0

            fg_shadow_dilate = cv2.dilate(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            fg_shadow_erode = cv2.erode(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            fg_shadow_edge = fg_shadow_dilate - fg_shadow_erode
            fg_shadow_edge = Image.fromarray(np.uint8(fg_shadow_edge), mode='L')


            #####erode foreground mask to produce synthetic image with smooth edge
            if len(instance_pixels) == 1:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((20, 20), np.uint8), iterations=1)
            elif len(instance_pixels) < 3:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            else:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((5, 5), np.uint8), iterations=1)
            fg_shadow_add = fg_shadow_new + new_shadow_mask
            fg_shadow_new[fg_shadow_add != 510] == 0


            shadow_object_ratio = np.sum(fg_shadow/255) / np.sum(fg_instance/255)
            whole_area = np.ones(np.shape(fg_shadow))
            shadow_ratio = np.sum(fg_shadow/255) / np.sum(whole_area)
            if is_train:
                ## selection
                if shadow_ratio < 0.002:
                    continue

            fg_instance = Image.fromarray(np.uint8(fg_instance), mode='L')
            fg_shadow = Image.fromarray(np.uint8(fg_shadow), mode='L')
            birdy_fg_instances.append(fg_instance)
            birdy_fg_shadows.append(fg_shadow)
            birdy_instance_boxes.append(torch.IntTensor(np.array(fg_instance_boxes)))
            birdy_shadow_boxes.append(torch.IntTensor(np.array(fg_shadow_boxes)))
            birdy_im_lists.append(imname_list)

            ####obtaining bbox area of foreground object
            fg_instance_box_areas = np.zeros(np.shape(fg_shadow))
            fg_shadow_box_areas = np.zeros(np.shape(fg_shadow))
            for i in range(len(fg_instance_boxes)):
                fg_instance_box_areas = bbox_to_mask(fg_instance_boxes[i],fg_instance_box_areas)
                fg_shadow_box_areas = bbox_to_mask(fg_shadow_boxes[i],fg_shadow_box_areas)
            fg_instance_box_areas = Image.fromarray(np.uint8(fg_instance_box_areas),mode='L')
            fg_shadow_box_areas = Image.fromarray(np.uint8(fg_shadow_box_areas),mode='L')
            birdy_shadow_box_areas.append(fg_shadow_box_areas)
            birdy_instance_box_areas.append(fg_instance_box_areas)



            new_shadow_free_image = deshadowed_image * (np.tile(np.expand_dims(np.array(fg_shadow_new) / 255, -1), (1, 1, 3))) + \
                                    shadow_image * (1 - np.tile(np.expand_dims(np.array(fg_shadow_new) / 255, -1),
                                                                (1, 1, 3)))

            birdy_deshadoweds.append(Image.fromarray(np.uint8(new_shadow_free_image), mode='RGB'))
            birdy_shadoweds.append(Image.fromarray(np.uint8(shadow_image), mode='RGB'))

            bg_instance = Image.fromarray(np.uint8(bg_instance),mode='L')
            bg_shadow = Image.fromarray(np.uint8(bg_shadow), mode='L')
            birdy_bg_shadows.append(bg_shadow)
            birdy_bg_instances.append(bg_instance)

            birdy_shadowparas.append(shadow_param)
            birdy_edges.append(fg_shadow_edge)
            birdy_shadow_object_ratio.append(shadow_object_ratio)
            fg_instance = []
            fg_shadow = []
            bg_instance = []
            bg_shadow = []
            fg_shadow_add = []
    return birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows,  birdy_bg_instances, \
           birdy_bg_shadows,birdy_edges, birdy_shadowparas, birdy_shadow_object_ratio, birdy_instance_boxes, birdy_shadow_boxes, birdy_instance_box_areas, birdy_shadow_box_areas, birdy_im_lists



class ShadowParamDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.is_train = self.opt.isTrain
        self.root = opt.dataroot
        self.dir_A =  opt.shadowimg_path 
        self.dir_C = opt.shadowfree_path 
        self.dir_param = opt.param_path
        self.dir_bg_instance = opt.bg_instance_path
        self.dir_bg_shadow = opt.bg_shadow_path
        self.dir_new_mask = opt.new_mask_path

        self.imname_total = []
        self.imname = []
        if self.is_train:
            for f in open(opt.dataroot + 'Training_labels.txt'):
                self.imname.append(f.split())
        else:
            for f in open(opt.dataroot + 'Testing_labels.txt'):
                self.imname_total.append(f.split())

            for im in self.imname_total:
                instance = Image.open(os.path.join(self.dir_bg_instance,im[0])).convert('L')
                instance = np.array(instance)
                instance_pixels = np.unique(np.sort(instance[instance>0]))
                shadow = Image.open(os.path.join(self.dir_bg_shadow,im[0])).convert('L')
                shadow = np.array(shadow)
                shadow_pixels = np.unique(np.sort(shadow[shadow>0]))
                if self.is_train:
                    self.imname = self.imname_total
                else:
                    # select bosfree image or bos image
                    if self.opt.bos:
                        if (len(instance_pixels) > 1):
                            self.imname.append(im)
                    elif self.opt.bosfree:
                        if (len(instance_pixels) == 1):
                            self.imname.append(im)

        # print('total images number', len(self.imname))
        self.birdy_deshadoweds = []
        self.birdy_shadoweds = []
        self.birdy_fg_instances = []
        self.birdy_fg_shadows = []
        self.birdy_bg_instances = []
        self.birdy_bg_shadows = []
        self.birdy_edges = []
        self.birdy_shadow_params = []
        self.birdy_shadow_object_ratio = []
        self.birdy_instance_boxes = []
        self.birdy_shadow_boxes= []
        self.birdy_instance_box_areas=[]
        self.birdy_shadow_box_areas=[]
        self.birdy_imlists=[]
        for imname_list in self.imname:
            imname = imname_list[0]
            A_img = Image.open(os.path.join(self.dir_A,imname)).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            C_img = Image.open(os.path.join(self.dir_C,imname)).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            new_mask = Image.open(os.path.join(self.dir_new_mask,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            instance = Image.open(os.path.join(self.dir_bg_instance,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            shadow = Image.open(os.path.join(self.dir_bg_shadow,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            imlist = imname_list
            sparam = open(os.path.join(self.dir_param,imname+'.txt'))
            line = sparam.read()
            shadow_param = np.asarray([float(i) for i in line.split(" ") if i.strip()])
            shadow_param = shadow_param[0:6]
            
            A_img_array = np.array(A_img)
            C_img_arry = np.array(C_img)
            new_mask_array = np.array(new_mask)
            instance_array = np.array(instance)
            shadow_array = np.array(shadow)

            ####object numbers
            instance_pixels = np.unique(np.sort(instance_array[instance_array>0]))
            object_num = len(instance_pixels)

            #####selecting random number of objects as foreground objects, while only one object is selected as foreground object
            self.birdy_deshadoweds, self.birdy_shadoweds,  self.birdy_fg_instances, self.birdy_fg_shadows, \
            self.birdy_bg_instances,  self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_shadow_object_ratio, \
            self.birdy_instance_boxes, self.birdy_shadow_boxes, self.birdy_instance_box_areas, self.birdy_shadow_box_areas, self.birdy_imlists = generate_training_pairs( \
                self.opt.loadSize, A_img_array, C_img_arry, instance_array, shadow_array, new_mask_array, shadow_param,imname_list, self.is_train, \
                self.birdy_deshadoweds, self.birdy_shadoweds,  self.birdy_fg_instances, self.birdy_fg_shadows, \
                self.birdy_bg_instances,  self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_shadow_object_ratio, \
                self.birdy_instance_boxes, self.birdy_shadow_boxes, self.birdy_instance_box_areas, self.birdy_shadow_box_areas,self.birdy_imlists)

           
        self.data_size = len(self.birdy_deshadoweds)
        # print('fff', self.is_train)
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
        birdy['A'] = self.birdy_shadoweds[index]
        birdy['C'] = self.birdy_deshadoweds[index]
        birdy['edge'] = self.birdy_edges[index]
        birdy['instancemask'] = self.birdy_fg_instances[index]
        birdy['B'] = self.birdy_fg_shadows[index]
        birdy['bg_shadow'] = self.birdy_bg_shadows[index]
        birdy['bg_instance'] = self.birdy_bg_instances[index]
        birdy['fg_instance_box_area'] = self.birdy_instance_box_areas[index]
        birdy['fg_shadow_box_area'] = self.birdy_shadow_box_areas[index]
        birdy['im_list']= self.birdy_imlists[index]


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
                if k=='im_list':
                    continue
                birdy[k] = im.resize((neww, newh),Image.NEAREST)

        if self.opt.no_flip and self.opt.no_crop and self.opt.no_rotate:
            for k,im in birdy.items():
                if k=='im_list':
                    continue
                birdy[k] = im.resize((neww, newh),Image.NEAREST)

        #### flip
        if not self.opt.no_flip:
            for i in ['A', 'B', 'C',  'instancemask', 'bg_shadow', 'bg_instance', 'edge','fg_instance_box', 'fg_shadow_box', 'fg_instance_box_area', 'fg_shadow_box_area']:
                birdy[i] = birdy[i].transpose(Image.FLIP_LEFT_RIGHT)


        for k,im in birdy.items():
            if k=='im_list':
                continue
            birdy[k] = self.transformB(im)


        for i in ['A','C','B','instancemask', 'bg_instance', 'bg_shadow', 'edge','fg_instance_box', 'fg_shadow_box', 'fg_instance_box_area', 'fg_shadow_box_area']:
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
            if k=='im_list':
                continue
            im = F.interpolate(im.unsqueeze(0), size = self.opt.loadSize)
            birdy[k] = im.squeeze(0)


        birdy['w'] = ow
        birdy['h'] = oh

        #if the shadow area is too small, let's not change anything:
        shadow_param = self.birdy_shadow_params[index]
        if torch.sum(birdy['B']>0) < 30 :
            shadow_param=[0,1,0,1,0,1]

        birdy['param'] = torch.FloatTensor(np.array(shadow_param))


        return birdy
    def __len__(self):
        return self.data_size

    def name(self):
        return 'ShadowParamDataset'
