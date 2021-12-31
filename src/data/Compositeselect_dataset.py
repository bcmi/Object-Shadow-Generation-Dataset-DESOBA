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

def transparent_png(img,fg_mask):
    img = img.convert('RGBA')
    W, H = img.size
    # fg_mask = np.array(fg_mask)
    for h in range(H):   ###循环图片的每个像素点
        for l in range(W):
            if fg_mask.getpixel((l,h)) == 0:
                pixels = img.getpixel((l,h))
                pixels = pixels[:-1] + (80,)
                img.putpixel((l,h),pixels)
    img.save('/media/user/data/ShadowGeneration/HYShadowGeneration/SOBAMixFGAllData/Model_SelfAttention_GRESNEXT18_C32_Dpixel_lrD0.0002/SelfAttention_Illumination1_Residual0_ConditionD1_Llight0_Lpara1_Lshadowrecons10_Limagerecons10_Lgan0.1_Lstn0_Nwarp0_Lref1_Ltv0_Lganmask0/SelfAttention_GRESNEXT18_C32_Dpixel_lrD0.0002/web/images/1.png')
    # img = Image.open('/media/user/data/ShadowGeneration/HYShadowGeneration/SOBAMixFGAllData/Model_SelfAttention_GRESNEXT18_C32_Dpixel_lrD0.0002/SelfAttention_Illumination1_Residual0_ConditionD1_Llight0_Lpara1_Lshadowrecons10_Limagerecons10_Lgan0.1_Lstn0_Nwarp0_Lref1_Ltv0_Lganmask0/SelfAttention_GRESNEXT18_C32_Dpixel_lrD0.0002/web/images/0.png')
    # print(img.size())
    return  img

def generate_training_pairs(shadow_image, deshadowed_image, instance_mask, shadow_mask, new_shadow_mask, shadow_param, imname,is_train, \
                            birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows,  birdy_bg_instances,  birdy_bg_shadows, birdy_edges, birdy_shadowparas, birdy_new_shadow_masks, birdy_max_objects, birdy_names):

    ####curret_image_object_selection_choice
    instance_pixels = np.unique(np.sort(instance_mask[instance_mask>0]))
    shadow_pixels = np.unique(np.sort(shadow_mask[shadow_mask>0]))
    all_instance_pixels = np.intersect1d(instance_pixels,shadow_pixels)

    ##calculatiiiing instance mask area, selecting suitable object
    instance_area_ratios=[]
    instance_pixels = []
    for pixel in all_instance_pixels:
        instance_ratio = np.sum(instance_mask[instance_mask==pixel]/255)/np.sum(np.ones(np.shape(instance_mask)))
        instance_area_ratios.append(instance_ratio)
        # setting restriction for one foreground object
        if instance_ratio>0.01 and instance_ratio<0.2:
            instance_pixels.append(pixel)

        # setting restriction on two foreground objects
        # if instance_ratio>0.005 and instance_ratio<0.2:
        #     instance_pixels.append(pixel)
    if len(instance_pixels)<1:
        return birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows,  birdy_bg_instances,  birdy_bg_shadows,birdy_edges, birdy_shadowparas, birdy_new_shadow_masks, birdy_max_objects


    object_num = len(instance_pixels)

    if not is_train:
        object_num += 1

    areas = []
    for pixel in instance_pixels:
        area = (instance_mask == pixel).sum()
        areas.append(area)
    max_area_objects = max(areas)
    max_index = areas.index(max_area_objects)
    bg_max_instance = instance_mask.copy()
    bg_max_instance[bg_max_instance!=instance_pixels[max_index]] = 0
    bg_max_instance[bg_max_instance==instance_pixels[max_index]] = 255



    #####selecting random number of objects as foreground objects, while only one object is selected as foreground object
    for i in range(1, object_num):
        selected_instance_pixel_combine = itertools.combinations(instance_pixels, i)
        combines = [combine for combine in selected_instance_pixel_combine]


        if not is_train:
            #####combination
            # if i!=1:
            #     continue
            #####select two foreground object
            if i!=2:
                continue
            # else:
            #     if len(combines) >2:
            #         combines = combines[:2]
        ######dealing with fg and bg
        j = -1
        for combine in combines:
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

            #fg_object size
            if (np.sum(fg_instance/255)/np.sum(np.ones(np.shape(fg_instance)))) < 0.05:
                continue

            # else:
            #     bbox = fg_instance
            #     x, y = bbox.nonzero()
            #     row_length = max(y) - min(y)
            #     col_length = max(x) - min(x)
            #     w = np.shape(fg_instance)[0]
            #     h = np.shape(fg_instance)[1]
            #     if row_length > 1/3 * w  or row_length > 1/3 * h:
            #         continue
            j+=1
            if j >1:
                break
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
            # birdy_max_objects.append(max_area_objects)
            birdy_max_objects.append(bg_max_instance)
            birdy_names.append(imname)

            fg_instance = []
            fg_shadow = []
            bg_instance = []
            bg_shadow = []
            fg_shadow_add = []
            # break
    return birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows,  birdy_bg_instances,  birdy_bg_shadows,birdy_edges, birdy_shadowparas, birdy_new_shadow_masks, birdy_max_objects, birdy_names









class CompositeselectDataset(BaseDataset):
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

        ##compostite image path
        self.dir_com_bg = opt.com_bg_path
        self.dir_com_fgmask = opt.com_fgmask_path

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
                if (len(instance_pixels) > 0):
                    self.imname.append(im)
                    continue

                # # more than one bg pair(126)
                # if (len(instance_pixels) > 1):
                #     self.imname.append(im)
                #     continue

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


        self.birdy_shadowimg_whole = []
        self.birdy_instances_whole = []
        self.birdy_shadows_whole = []
        self.birdy_max_instance_area_whole = []


        self.birdy_names = []

        ## shadow immage with image name: according name to recall correspondng shadow image
        self.birdy_shadowimg_whole_originsize = []
        self.birdy_originsize_imname = []
        ## shadow immage with image name
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

            self.birdy_shadowimg_whole.append(A_img)
            self.birdy_instances_whole.append(instance)
            self.birdy_shadows_whole.append(shadow)
            self.birdy_shadowimg_whole_originsize.append(Image.open(os.path.join(self.dir_A,imname)).convert('RGB'))
            self.birdy_originsize_imname.append(imname)

            A_img_array = np.array(A_img)
            C_img_arry = np.array(C_img)
            new_mask_array = np.array(new_mask)
            instance_array = np.array(instance)
            shadow_array = np.array(shadow)

            ####object numbers
            instance_pixels = np.unique(np.sort(instance_array[instance_array>0]))
            object_num = len(instance_pixels)
            areas = []
            for pixel in instance_pixels:
                area = (instance_array == pixel).sum()
                areas.append(area)
            max_area_objects = max(areas)
            self.birdy_max_instance_area_whole.append(max_area_objects)

       
            #####selecting random number of objects as foreground objects, while only one object is selected as foreground object
            self.birdy_deshadoweds, self.birdy_shadoweds,  self.birdy_fg_instances, self.birdy_fg_shadows,  self.birdy_bg_instances, \
            self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_new_shadow_masks, self.birdy_max_objects, self.birdy_names = generate_training_pairs( \
                A_img_array, C_img_arry, instance_array, shadow_array, new_mask_array, shadow_param, imname,self.is_train, \
                self.birdy_deshadoweds, self.birdy_shadoweds,  self.birdy_fg_instances, self.birdy_fg_shadows, \
                self.birdy_bg_instances,  self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_new_shadow_masks, self.birdy_max_objects, self.birdy_names)

            
            # one_foreground_object
            self.birdy_com_name = []
            self.birdy_com_shadowimg = []
            self.birdy_com_fgmask = []
            # self.birdy_shadows_orin_whole, self.birdy_com_shadowimg, self.birdy_com_fgmask
            self.birdy_shadows_orin_whole = []
            for index in range(76):
                img_name = '{}.png'.fromat(index)
                com_shadowimg = Image.open(os.path.join(self.dir_com_bg,img_name)).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
                com_fgmask = Image.open(os.path.join(self.dir_com_fgmask,imname)).convert('L').resize((self.opt.loadSize, self.opt.loadSize),Image.NEAREST)
                for j in range(len(self.birdy_shadowimg_whole)):
                    if np.sum((np.array(birdy['com_shadowfree']) - self.birdy_shadowimg_whole[j]) * (1-np.array(birdy['com_fgmask'])/255))==0:
                        self.birdy_shadows_orin_whole.append(self.birdy_shadowimg_whole_originsize[j])
                        self.birdy_com_shadowimg.append(com_shadowimg)
                        self.birdy_com_fgmask.append(com_fgmask)
                        self.birdy_com_name.append(self.birdy_originsize_imname[j])
                    else:
                        break



        self.data_size = len(self.birdy_deshadoweds)
        # print('fff', self.is_train)
        print('datasize', self.data_size)
        print('bg nums',len(self.birdy_shadows_whole))
        print('composite num',len(self.birdy_shadows_orin_whole))

        self.transformB = transforms.Compose([transforms.ToTensor()])

        self.transformAugmentation = transforms.Compose([
            transforms.Resize(int(self.opt.loadSize * 1.12), Image.BICUBIC),
            transforms.RandomCrop(self.opt.loadSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    def __getitem__(self,index):
        #####background
        # composite image
        ##unfixed
        ###replace fg in composite image with original foreground object
        # self.birdy_shadows_orin_whole, self.birdy_com_shadowimg, self.birdy_com_fgmask
        for com_index in range(len(self.birdy_com_name)):
            orin_img = self.birdy_shadows_orin_whole[com_index]
            com_img = self.birdy_com_shadowimg[com_index]
            com_fgmask = self.birdy_com_fgmask[com_index]
            #match foreground
            if np.sum(np.array(com_fgmask)) = np.sum(np.array(self.birdy_fg_instances[index])):
                fg_mask_name = self.birdy_names[index]

                origin_index = self.birdy_originsize_imname.index(fg_mask_name)
                origin_fg_shadowimage = self.birdy_shadowimg_whole_originsize[origin_index]
                


                bg_name = self.birdy_com_name[com_index]
                origin_bg_index = self.birdy_originsize_imname.index(bg_name)
                origin_bg_shadowimage = self.birdy_shadowimg_whole_originsize[origin_bg_index]



                ##fg_mask in origin bg image position
                ow = origin_bg_shadowimage.size[0]
                oh = origin_bg_shadowimage.size[1]

                #resize composite image
                resize_com = com_img.resize((ow,oh),Image.NEAREST)
                resize_fg_mask = com_fgmask.resize((ow,oh),Image.NEAREST)
                # position of fg in composite image
                foreground_object = np.array(self.birdy_fg_instances[index])
                bbox = foreground_object
                x, y = bbox.nonzero()
                

                #resize fgimg
                resize_origin_fg_shadowimg = origin_fg_shadowimage.resize((ow,oh),Image.NEAREST)
                resize_origin_fg_mask = self.birdy_fg_instances[index].resize((ow,oh),Image.NEAREST)
                birdy['bg_shadow'] =  self.birdy_shadows_whole[index_bg].resize((ow,oh),Image.NEAREST)
                birdy['bg_instance'] = self.birdy_instances_whole[index_bg].resize((ow,oh),Image.NEAREST)

                # position of fg in original fg image
                bbox_o = np.array(resize_origin_fg_mask)
                x_o, y_o = bbox.nonzero()
                row_length_o = max(y_o) - min(y_o)
                col_length_o = max(x_o) - min(x_o)

                #new image
                origin_bg_shadowimage[min(x):max(x), min(y):max(y)] = resize_origin_fg_shadowimg[min(x_o):max(x_o), min(y_o):max(y_o)]
                birdy['instancemask'] = Image.fromarray(np.uint8(foreground_object), mode='L') 
                birdy['C'] = origin_bg_shadowimage

                # birdy_new['C'] = Image.fromarray(np.uint8(final_img), mode='RGB')
                for k,im in birdy.items():
                    if k == 'max_object' or k=='ratio':
                        continue
                    birdy[k] = self.transformB(im)


                for k,im in birdy.items():
                    birdy[k] = (im - 0.5)*2



                h = birdy['A'].size()[1]
                w = birdy['A'].size()[2]


                # for k,im in birdy.items():
                #     if k == 'max_object' or k == 'ratio':
                #         continue
                #     im = F.interpolate(im.unsqueeze(0), size = self.opt.loadSize)
                #     birdy[k] = im.squeeze(0)

                birdy['w'] = ow
                birdy['h'] = oh


        return birdy

    def __len__(self):
        return self.data_size

    def name(self):
        return 'Compositeselect'
