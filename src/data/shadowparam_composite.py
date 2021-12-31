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


is_test_finetune = False

######'dir_A': shadowimage
######'dir_B': shadowmask
######'dir_C': shadowfree
######'dir_param':illumination parameter
######'dir_light': light direction
######'dir_instance':object mask


def generate_training_pairs(shadow_image, deshadowed_image, instance_mask, shadow_mask, new_shadow_mask, shadow_param, is_train, \
                            birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows,  birdy_bg_instances,  birdy_bg_shadows, birdy_edges, birdy_shadowparas, birdy_new_shadow_masks, birdy_max_objects):

    ####curret_image_object_selection_choice
    instance_pixels = np.unique(np.sort(instance_mask[instance_mask>0]))
    object_num = len(instance_pixels)

    areas = []
    for pixel in instance_pixels:
        area = (instance_mask == pixel).sum()
        areas.append(area)
    max_area_objects = max(areas)



    #####selecting random number of objects as foreground objects, while only one object is selected as foreground object
    for i in range(1, object_num):
        selected_instance_pixel_combine = itertools.combinations(instance_pixels, i)
        if not is_train:
            #####combination
            if i!=1:
                continue
        ######dealing with fg and bg
        for combine in selected_instance_pixel_combine:
            fg_instance = instance_mask.copy()
            fg_shadow = shadow_mask.copy()
            bg_instance = instance_mask.copy()
            bg_shadow = shadow_mask.copy()
            fg_shadow[fg_shadow==255] = 0
            remaining_fg_pixel = list(set(instance_pixels).difference(set(combine)))
            for pixel in combine:
                fg_shadow[fg_shadow==pixel] = 255
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


            #####erode foreground mask birdy['B']
            if len(instance_pixels) == 1:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((20, 20), np.uint8), iterations=1)
            elif len(instance_pixels) < 3:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            else:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((5, 5), np.uint8), iterations=1)
            fg_shadow_add = fg_shadow_new + new_shadow_mask
            fg_shadow_new[fg_shadow_add != 510] == 0

            fg_instance = Image.fromarray(np.uint8(fg_instance), mode='L')
            fg_shadow = Image.fromarray(np.uint8(fg_shadow), mode='L')
            birdy_fg_instances.append(fg_instance)
            birdy_fg_shadows.append(fg_shadow)

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
            new_shadow_mask = Image.fromarray(np.uint8(new_shadow_mask),mode='L')
            birdy_new_shadow_masks.append(new_shadow_mask)
            birdy_max_objects.append(max_area_objects)
            fg_instance = []
            fg_shadow = []
            bg_instance = []
            bg_shadow = []
            fg_shadow_add = []
    return birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows,  birdy_bg_instances,  birdy_bg_shadows,birdy_edges, birdy_shadowparas, birdy_new_shadow_masks, birdy_max_objects









class ShadowParamDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.is_train = self.opt.isTrain
        self.root = opt.dataroot
        self.dir_A =  opt.shadowimg_path #os.path.join(opt.dataroot, 'A')
        self.dir_C = opt.shadowfree_path #os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_param = opt.param_path
        self.dir_bg_instance = opt.bg_instance_path
        self.dir_bg_shadow = opt.bg_shadow_path
        self.dir_new_mask = opt.new_mask_path

        self.imname_total = []
        self.imname = []
        if self.is_train:
            for f in open(opt.dataroot + 'Training_labels.txt'):
                self.imname_total.append(f.split())
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
                ########selecting testing conditional images
                ####total(160)
                # # more than one bg pair(126)
                if (len(instance_pixels) > 1):
                    self.imname.append(im)
                    continue

                # # only shadow(10) + no bg information (24)
                # if (len(instance_pixels) == 1):
                #     self.imname.append(im)
                #     continue

                ##only shadow(10)
                # if (len(instance_pixels) == 1 and len(shadow_pixels)>1):
                #     self.imname.append(im)
                #     continue

                # ##no bg information (24)
                # if (len(shadow_pixels) == 1 and len(instance_pixels) == 1):
                #     self.imname.append(im)
                #     continue
                ########selecting testing conditional images
        self.birdy_deshadoweds = []
        self.birdy_shadoweds = []
        self.birdy_fg_instances = []
        self.birdy_fg_shadows = []
        self.birdy_bg_instances = []
        self.birdy_bg_shadows = []
        self.birdy_edges = []
        self.birdy_shadow_params = []
        self.birdy_new_shadow_masks = []
        self.birdy_max_objects = []
        for imname_list in self.imname:
            imname = imname_list[0]
            A_img = Image.open(os.path.join(self.dir_A,imname)).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            C_img = Image.open(os.path.join(self.dir_C,imname)).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            new_mask = Image.open(os.path.join(self.dir_new_mask,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            instance = Image.open(os.path.join(self.dir_bg_instance,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            shadow = Image.open(os.path.join(self.dir_bg_shadow,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
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
            self.birdy_deshadoweds, self.birdy_shadoweds,  self.birdy_fg_instances, self.birdy_fg_shadows,  self.birdy_bg_instances, \
            self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_new_shadow_masks, self.birdy_max_objects = generate_training_pairs( \
                A_img_array, C_img_arry, instance_array, shadow_array, new_mask_array, shadow_param, self.is_train, \
                self.birdy_deshadoweds, self.birdy_shadoweds,  self.birdy_fg_instances, self.birdy_fg_shadows, \
                self.birdy_bg_instances,  self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_new_shadow_masks, self.birdy_max_objects)







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
        is_samebg_difffg = False
        ### same bg, different fg
        if is_samebg_difffg:
            ######foreground
            birdy = {}
            birdy['AA'] = self.birdy_shadoweds[index]
            birdy['edge'] = self.birdy_edges[index]
            birdy['instancemask'] = self.birdy_fg_instances[index]
            birdy['B'] = self.birdy_fg_shadows[index]
            birdy['new_shadow_mask'] = self.birdy_new_shadow_masks[index]

            #####background
            # index_bg = index  images: 102, 116
            # index_bg = 110  images:146
            # index_bg = 191 images:447
            # index_bg = 271 images:145
            index_bg = 110
            birdy['A'] = self.birdy_shadoweds[index_bg]
            birdy['bg_shadow'] =  Image.fromarray(np.uint8(np.array(self.birdy_bg_shadows[index_bg]) + np.array(self.birdy_fg_shadows[index_bg])), mode='L')
            birdy['bg_instance'] = Image.fromarray(np.uint8(np.array(self.birdy_bg_instances[index_bg]) + np.array(self.birdy_fg_instances[index_bg])), mode='L')
            birdy['light_image'] = birdy['A']

        else:
            ### same fg, different bg
            ######foreground
            index_fg = 102
            birdy = {}
            birdy['AA'] = self.birdy_shadoweds[index_fg]
            birdy['edge'] = self.birdy_edges[index_fg]
            birdy['instancemask'] = self.birdy_fg_instances[index_fg]
            birdy['B'] = self.birdy_fg_shadows[index_fg]
            birdy['new_shadow_mask'] = self.birdy_new_shadow_masks[index_fg]

            #####background
            # index_bg = index  images: 102, 116
            # index_bg = 110  images:146
            # index_bg = 191 images:447
            # index_bg = 271 images:145

            birdy['A'] = self.birdy_shadoweds[index]
            birdy['bg_shadow'] =  Image.fromarray(np.uint8(np.array(self.birdy_bg_shadows[index]) + np.array(self.birdy_fg_shadows[index])), mode='L')
            birdy['bg_instance'] = Image.fromarray(np.uint8(np.array(self.birdy_bg_instances[index]) + np.array(self.birdy_fg_instances[index])), mode='L')
            birdy['light_image'] = birdy['A']


        bg_instance = np.array(birdy['bg_instance'])
        bg_shadow = np.array(birdy['bg_shadow'])


        # ratio =  int(birdy['max_object'] / np.sum(np.array(birdy['instancemask'])/255))




        ####erode to acquire more available area
        bg_shadow = cv2.dilate(bg_shadow, np.ones((10, 10), np.uint8), iterations=1)
        bg_instance = cv2.dilate(bg_instance, np.ones((10, 10), np.uint8), iterations=1)
        pure_bg_area = bg_instance.copy()
        pure_bg_area[bg_instance>0] = 255
        pure_bg_area[bg_shadow>0] = 255
        pure_bg_area[pure_bg_area==0] = 1
        pure_bg_area[pure_bg_area==255] = 0
        birdy['pure_bg_area'] = Image.fromarray(np.uint8(pure_bg_area), mode='L')



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

        ###same bg, different fg
        if is_samebg_difffg:
            birdy['max_object'] = self.birdy_max_objects[index_bg]
        ###same fg, different bg
        else:
            birdy['max_object'] = self.birdy_max_objects[index]
        ratio =  birdy['max_object'] / np.sum(np.array(birdy['instancemask'])/255)
        birdy['ratio'] = ratio


        ####composite images
        ####three-dimensional fg instance
        A_img_foreground = np.array(birdy['AA'])
        fg_instance = np.array( birdy['instancemask'] )
        fg_shadow = np.array( birdy['B'] )
        fg_instance_3_dim = np.tile(np.expand_dims(fg_instance,2), (1,1,3))

        foreground_object = np.array(birdy['instancemask'])

        # print('ratio', ratio)
        # if ratio>1:
        #     # ratio_pixel = int((ratio -1) * (np.sum(np.array(birdy['instancemask'])/255)))
        #     # print('ratio_pixel',ratio_pixel )
        #     ratio_pixel = 1
        #     foreground_object = cv2.erode(foreground_object, np.ones((ratio_pixel, ratio_pixel), np.uint8), iterations=1)
        # else:
        #     # ratio_pixel = int(ratio  * (np.sum(np.array(birdy['instancemask'])/255)))
        #     # print('ratio_pixel',ratio_pixel )
        #     ratio_pixel = 3
        #     foreground_object = cv2.dilate(foreground_object, np.ones((ratio_pixel, ratio_pixel), np.uint8), iterations=1)
        bbox = foreground_object
        x, y = bbox.nonzero()
        row_length = max(y) - min(y)
        col_length = max(x) - min(x)

        for i in range(self.opt.loadSize-20,0,-1):
            # for i in range(self.opt.loadSize-20):
            x_offset = int( (self.opt.loadSize - col_length - 1)/self.opt.loadSize * (i + 10) )
            y_offset = int( (self.opt.loadSize - row_length - 1)/self.opt.loadSize * (i + 10) )
            new_foreground_object = np.zeros(np.shape(birdy['A']))
            new_foreground_object_mask = np.zeros(np.shape(birdy['bg_instance']))
            new_foreground_shadow_mask = np.zeros(np.shape(birdy['bg_instance']))

            new_foreground_object_mask[x_offset:x_offset + col_length, y_offset:y_offset+row_length] = fg_instance[min(x):max(x), min(y):max(y)]
            new_foreground_shadow_mask[x_offset:x_offset + col_length, y_offset:y_offset+row_length] = fg_shadow[min(x):max(x), min(y):max(y)]

            new_foreground_object_mask[new_foreground_object_mask>0] = 255
            new_foreground_shadow_mask[new_foreground_shadow_mask>0] = 255

            #####erode according ratio
            # if ratio>1:
            #     # ratio_pixel = int((ratio -1) * (np.sum(np.array(birdy['instancemask'])/255)))
            #     # print('ratio_pixel',ratio_pixel )
            #     ratio_pixel = 1
            #     new_foreground_object_mask = cv2.erode(new_foreground_object_mask, np.ones((ratio_pixel, ratio_pixel), np.uint8), iterations=1)
            #     new_foreground_shadow_mask = cv2.erode(new_foreground_shadow_mask, np.ones((ratio_pixel, ratio_pixel), np.uint8), iterations=1)
            # else:
            #     # ratio_pixel = int(ratio * (np.sum(np.array(birdy['instancemask'])/255)))
            #     # print('ratio_pixel',ratio_pixel )
            #     ratio_pixel = 3
            #     new_foreground_object_mask = cv2.dilate(new_foreground_object_mask, np.ones((ratio_pixel, ratio_pixel), np.uint8), iterations=1)
            #     new_foreground_shadow_mask = cv2.dilate(new_foreground_shadow_mask, np.ones((ratio_pixel, ratio_pixel), np.uint8), iterations=1)


            new_foreground_object[x_offset:x_offset + col_length, y_offset:y_offset+row_length ,:] = A_img_foreground[min(x):max(x), min(y):max(y),:]

            overlap = (new_foreground_object_mask) * birdy['pure_bg_area']
            if np.sum(overlap) == np.sum(new_foreground_object_mask):
                if i < 40:
                    continue
                fg_instance = Image.fromarray(np.uint8(new_foreground_object_mask), mode='L')
                birdy['instancemask'] = fg_instance
                fg_shadow = Image.fromarray(np.uint8(new_foreground_shadow_mask), mode='L')
                birdy['B'] = fg_shadow
                ####replacing the area of foreground
                final_img = new_foreground_object * (np.tile(np.expand_dims(new_foreground_object_mask / 255, -1), (1, 1, 3))) + \
                            birdy['A'] * (1 - np.tile(np.expand_dims(np.array(new_foreground_object_mask) / 255, -1), (1, 1, 3)))
                birdy['C'] = Image.fromarray(np.uint8(final_img), mode='RGB')
                # birdy['C'] = Image.fromarray(np.uint8(new_foreground_object * (np.tile(np.expand_dims(new_foreground_object_mask / 255, -1), (1, 1, 3)))), mode='RGB')
                # birdy['C'] = Image.fromarray(np.uint8(A_img_foreground),mode='RGB')
                # birdy['A'] = birdy['AA']
                print('foreground object size', row_length, col_length)
                print('composite finish')
                break
            else:
                if i == self.opt.loadSize - 21:
                    birdy['instancemask'] = birdy['bg_instance']
                    birdy['C'] = birdy['A']
                    # birdy['A'] = birdy['AA']
                continue


        #### flip
        if not self.opt.no_flip:
            for i in ['A', 'B', 'C', 'light_image', 'instancemask', 'bg_shadow', 'bg_instance', 'edge']:
                birdy[i] = birdy[i].transpose(Image.FLIP_LEFT_RIGHT)

        # for i in ['A','C','B','instancemask', 'light_image', 'bg_instance', 'bg_shadow', 'edge']:
        #     if i in birdy:
        #         birdy[k] = self.transformB(birdy[k])

        for k,im in birdy.items():
            if k == 'max_object' or k=='ratio':
                continue
            birdy[k] = self.transformB(im)


        for i in ['A','C','B','instancemask', 'light_image', 'bg_instance', 'bg_shadow', 'edge']:
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
            if k == 'max_object' or k == 'ratio':
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
        birdy['light'] = birdy['param'][:4]
        return birdy
    def __len__(self):
        return self.data_size

    def name(self):
        return 'ShadowParamDataset'
