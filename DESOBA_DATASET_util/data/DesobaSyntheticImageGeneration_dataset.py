import os.path
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from PIL import Image
import torch
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

def Generation_Synthetic_Composite_Image(shadow_image, deshadowed_image, instance_mask, shadow_mask, is_train):
    fg_instances = [] 
    fg_shadows= []
    bg_instances= []
    bg_shadows= []
    synthetic_composite_images= []

    # calculating the number of objects
    ####producing training/test pairs according pixel value
    instance_pixels_a = np.unique(np.sort(instance_mask[instance_mask>0]))
    shadow_pixels_a = np.unique(np.sort(shadow_mask[shadow_mask>0]))
    pair_pixels = np.intersect1d(instance_pixels_a,shadow_pixels_a)
    object_num = len(pair_pixels)
    if object_num==1:
        object_num=2

    #####selecting random number of objects K as foreground objects
    for i in range(1, object_num):
        # Randomly selecting i foreground objects
        selected_instance_pixel_combine = itertools.combinations(pair_pixels, i)
        # Setting the number of foreground objects
        if is_train:
            # For training phase: we randomly select 1 or 2 foreground objects and replace their shadow areas with the counter parts in deshadowed image to obtain training synthetic composite image.
            if i>2:
                continue
        else:
            # For testing phase: we select 1 foreground object to and replace their shadow areas with the counter parts in deshadowed image to obtain testing synthetic composite image.
            if i != 1:
                continue

        # Replacing the shadow areas of selected foreground objects with the counter parts in deshadowed image to obtain foreground object mask, foreground shadow mask, background object mask, background shadow mask
        for combine in selected_instance_pixel_combine:
            fg_instance = instance_mask.copy()
            fg_shadow = shadow_mask.copy()
            # Removing shadow without corresponding objects from foreground shadow mask
            fg_shadow[fg_shadow==255] = 0
            # Setting the foreground shadow areas of selected foreground objects as 255, the remaining areas are set as 0
            for pixel in combine:
                fg_shadow[fg_shadow==pixel] = 255
                fg_instance[fg_instance==pixel] = 255
            fg_shadow[fg_shadow!=255] = 0
            fg_instance[fg_instance!=255] = 0


            # Finding the background objects relative to the selected foreground objects
            remaining_fg_pixel = list(set(pair_pixels).difference(set(combine)))
            bg_instance = instance_mask.copy()
            bg_shadow = shadow_mask.copy()
            # Setting the background shadow areas of background objects as 255, the remaining areas are set as 0
            # if Removing shadow without corresponding objects from background shadow mask, using bg_shadow[bg_shadow==255] = 0
            # bg_shadow[bg_shadow==255] = 0
            for pixel in remaining_fg_pixel:
                bg_shadow[bg_shadow==pixel]=255
                bg_instance[bg_instance==pixel]=255
            bg_shadow[bg_shadow!=255] = 0
            bg_instance[bg_instance!=255] = 0


            #To obtain realistic synthetic composite image with smooth edge, dilating the foreground shadow
            if len(pair_pixels) == 1:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((20, 20), np.uint8), iterations=1)
            elif len(pair_pixels) < 3:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            else:
                fg_shadow_new = cv2.dilate(fg_shadow, np.ones((5, 5), np.uint8), iterations=1)

            # Converting numpy array to grey image for foreground object, foreground shadow, background object, background shadow
            fg_instance = Image.fromarray(np.uint8(fg_instance), mode='L')
            fg_shadow = Image.fromarray(np.uint8(fg_shadow), mode='L')
            bg_instance = Image.fromarray(np.uint8(bg_instance),mode='L')
            bg_shadow = Image.fromarray(np.uint8(bg_shadow), mode='L')

            # Replacing the shadow areas of selected foreground objects with the counter parts in deshadowed image to obtain synthetic composite image with smooth edge
            synthetic_composite_image = deshadowed_image * (np.tile(np.expand_dims(np.array(fg_shadow_new) / 255, -1), (1, 1, 3))) + \
                                    shadow_image * (1 - np.tile(np.expand_dims(np.array(fg_shadow_new) / 255, -1),
                                                                (1, 1, 3)))
            synthetic_composite_image = Image.fromarray(np.uint8(synthetic_composite_image), mode='RGB')
            fg_instances.append(fg_instance) 
            fg_shadows.append(fg_shadow)
            bg_instances.append(bg_instance)
            bg_shadows.append(bg_shadow)
            synthetic_composite_images.append(synthetic_composite_image)

    return fg_instances, fg_shadows, bg_instances, bg_shadows, synthetic_composite_images




class DesobaSyntheticImageGenerationdataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.is_train = self.opt.isTrain
        self.root = opt.dataroot
        self.dir_A =  opt.shadowimg_path #os.path.join(opt.dataroot, 'A')
        self.dir_C = opt.shadowfree_path #os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_instance = opt.instance_path
        self.dir_shadow = opt.shadow_path

        self.store_path = opt.output_path

        self.imname_total = []
        self.imname = []

        self.shadow_imgs = []
        self.synthetic_composite_imgs = []
        self.fg_instance_masks = []
        self.fg_shadow_masks = []
        self.bg_instance_masks = []
        self.bg_shadow_masks = []

        self.bos = opt.bos
        self.bosfree = opt.bosfree

        self.transformB = transforms.Compose([transforms.ToTensor()])

        # according data split file to obtain training/test list
        if self.is_train:
            for f in open(opt.dataroot + 'Training_labels.txt'):
                self.imname_total.append(f.split())
        else:
            for f in open(opt.dataroot + 'Testing_labels.txt'):
                self.imname_total.append(f.split())

        
        for im in self.imname_total:
            if self.is_train:
                self.imname.append(im)
            else:
                instance = Image.open(os.path.join(self.dir_instance,im[0])).convert('L')
                instance_mask = np.array(instance)
                # Calculating the number of pairs
                pair_pixels = np.unique(np.sort(instance_mask[instance_mask>0]))
                # splitting the test images into BOS and BOS-free images, BOS: testing image pairs with background object-shadow, BOS-free: testing image pairs without background object-shadow
                if self.bos:
                    # selecting BOS testing images
                    if (len(pair_pixels) > 1):
                        self.imname.append(im)
                elif self.bosfree:
                    # selecting BOS-free testing images
                    if (len(pair_pixels) == 1):
                        self.imname.append(im)


        for imname_list in self.imname:
            imname = imname_list[0]
            A_img = Image.open(os.path.join(self.dir_A,imname)).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            C_img = Image.open(os.path.join(self.dir_C,imname)).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            instance = Image.open(os.path.join(self.dir_instance,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
            shadow = Image.open(os.path.join(self.dir_shadow,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)

            shadow_image = np.array(A_img)
            deshadowed_image = np.array(C_img)
            instance_mask = np.array(instance)
            shadow_mask = np.array(shadow)

            #Obtaining the number of objects
            object_num = len(np.unique(np.sort(instance_mask[instance_mask>0])))

            fg_instance, fg_shadow, bg_instance, bg_shadow, synthetic_composite_image = Generation_Synthetic_Composite_Image(shadow_image, deshadowed_image, instance_mask, shadow_mask, self.is_train)
            self.shadow_imgs.extend([A_img for e in range(len(synthetic_composite_image))])
            self.synthetic_composite_imgs.extend(synthetic_composite_image)
            self.fg_instance_masks.extend(fg_instance)
            self.fg_shadow_masks.extend(fg_shadow)
            self.bg_instance_masks.extend(bg_instance)
            self.bg_shadow_masks.extend(bg_shadow)
            


        self.data_size = len(self.synthetic_composite_imgs)
        self.image_size = len(self.imname)

        if self.is_train:
            print('producing {} train images with {} train pairs stored in {}'.format(self.image_size, self.data_size, self.store_path))
        else:
            if self.bos:
                print('producing {} bos test images with {} test bos pairs stored in {}'.format(self.image_size, self.data_size, self.store_path))
            if self.bosfree:
                print('producing {} bosfree test images with {} test bosfree pairs stored in {}'.format(self.image_size, self.data_size, self.store_path))

    def __getitem__(self,index):
        birdy = {}
        birdy['Synth_img'] = self.synthetic_composite_imgs[index]
        birdy['Shadow_img'] = self.shadow_imgs[index]
        birdy['fg_instance_mask'] = self.fg_instance_masks[index]
        birdy['fg_shadow_mask'] = self.fg_shadow_masks[index]
        birdy['bg_shadow_mask'] = self.bg_shadow_masks[index]
        birdy['bg_instance_mask'] = self.bg_instance_masks[index]


        ow, oh = birdy['Synth_img'].size[0], birdy['Synth_img'].size[1]
        loadSize = self.opt.loadSize
        neww = loadSize
        newh = loadSize



        for k,im in birdy.items():
            birdy[k] = im.resize((neww, newh),Image.NEAREST)
            birdy[k] = self.transformB(im)
            birdy[k] = (birdy[k] - 0.5)*2


        return birdy

    def __len__(self):
        return self.data_size

    def name(self):
        return 'ShadowGenerationDataset'

