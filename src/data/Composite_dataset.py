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



def generate_training_pairs(shadow_image, deshadowed_image, instance_mask, shadow_mask, new_shadow_mask, shadow_param, is_train, \
                            birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows,  birdy_bg_instances,  birdy_bg_shadows, birdy_edges, birdy_shadowparas, birdy_new_shadow_masks, birdy_max_objects):

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

            fg_instance = []
            fg_shadow = []
            bg_instance = []
            bg_shadow = []
            fg_shadow_add = []
            # break
    return birdy_deshadoweds, birdy_shadoweds,  birdy_fg_instances, birdy_fg_shadows,  birdy_bg_instances,  birdy_bg_shadows,birdy_edges, birdy_shadowparas, birdy_new_shadow_masks, birdy_max_objects









class CompositeDataset(BaseDataset):
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
                if (len(instance_pixels) > 0):
                    self.imname.append(im)
                    continue

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
            self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_new_shadow_masks, self.birdy_max_objects = generate_training_pairs( \
                A_img_array, C_img_arry, instance_array, shadow_array, new_mask_array, shadow_param, self.is_train, \
                self.birdy_deshadoweds, self.birdy_shadoweds,  self.birdy_fg_instances, self.birdy_fg_shadows, \
                self.birdy_bg_instances,  self.birdy_bg_shadows, self.birdy_edges, self.birdy_shadow_params, self.birdy_new_shadow_masks, self.birdy_max_objects)


        self.data_size = len(self.birdy_deshadoweds)
        print('bg nums',len(self.birdy_shadows_whole))

        self.transformB = transforms.Compose([transforms.ToTensor()])

        self.transformAugmentation = transforms.Compose([
            transforms.Resize(int(self.opt.loadSize * 1.12), Image.BICUBIC),
            transforms.RandomCrop(self.opt.loadSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    def __getitem__(self,index):
        list_birdy = []
        for index_bg  in  range(len(self.birdy_shadowimg_whole)):
            birdy = {}
            birdy['AA'] = self.birdy_shadoweds[index]
            birdy['instancemask'] = self.birdy_fg_instances[index]
            birdy['B'] = self.birdy_fg_shadows[index]
            birdy['new_shadow_mask'] = self.birdy_new_shadow_masks[index]

            birdy['A'] = self.birdy_shadowimg_whole[index_bg]
            birdy['bg_shadow'] =  self.birdy_shadows_whole[index_bg]
            birdy['bg_instance'] = self.birdy_instances_whole[index_bg]

            bg_instance = np.array(birdy['bg_instance'])
            bg_shadow = np.array(birdy['bg_shadow'])


            ####erode to acquire more available area
            bg_shadow = cv2.dilate(bg_shadow, np.ones((10, 10), np.uint8), iterations=1)
            bg_instance = cv2.dilate(bg_instance, np.ones((10, 10), np.uint8), iterations=1)
            pure_bg_area = bg_instance.copy()
            pure_bg_area[bg_instance>0] = 255
            pure_bg_area[bg_shadow>0] = 255
            pure_bg_area[pure_bg_area==0] = 1
            pure_bg_area[pure_bg_area==255] = 0
            birdy['pure_bg_area'] = Image.fromarray(np.uint8(pure_bg_area), mode='L')

            ##selecting foreground object compatible with background image
            fg_instance = np.array(birdy['instancemask'])
            fg_instance_area = np.sum(fg_instance[fg_instance>0]/255)
            bg_instance_pixels = np.unique(np.sort(bg_instance[bg_instance>0]))
            bg_instance_areas = []
            for pixel in bg_instance_pixels:
                instance_area = np.sum(bg_instance[bg_instance==pixel]/255)
                bg_instance_areas.append(instance_area)
            bg_instance_areas = sorted(bg_instance_areas)
            # one foreground
            # if fg_instance_area/bg_instance_areas[-1] > 1.2 or fg_instance_area/bg_instance_areas[-1] < 0.4:
            #     continue
            # # two foreground
            if fg_instance_area/bg_instance_areas[-1] > 2.4 or fg_instance_area/bg_instance_areas[-1] < 1.2:
                continue

            # if len(bg_instance_areas)==1:
            #     if fg_instance_area/bg_instance_areas[0] > 1.3 or fg_instance_area/bg_instance_areas[0]<0.7:
            #         continue
            # else:
            #     if fg_instance_area/bg_instance_areas[-1] > 1.3 or fg_instance_area/bg_instance_areas[-1]<0.7:
            #         continue

            ow = birdy['A'].size[0]
            oh = birdy['A'].size[1]
            loadSize = self.opt.loadSize

            neww = loadSize
            newh = loadSize

            if not self.is_train:
                for k,im in birdy.items():
                    birdy[k] = im.resize((neww, newh),Image.NEAREST)

            ### foreground object / max(background object)
            birdy['max_object'] = self.birdy_max_instance_area_whole[index_bg]
            ratio =   np.sum(np.array(birdy['instancemask'])/255)  /  np.sum(birdy['max_object']/255)
            birdy['ratio'] = ratio




            ####composite images
            ####three-dimensional fg instance
            A_img_foreground = np.array(birdy['AA'])
            fg_instance = np.array( birdy['instancemask'] )
            fg_shadow = np.array( birdy['B'] )
            fg_instance_3_dim = np.tile(np.expand_dims(fg_instance,2), (1,1,3))
            foreground_object = np.array(birdy['instancemask'])

            bbox = foreground_object
            x, y = bbox.nonzero()
            row_length = max(y) - min(y)
            col_length = max(x) - min(x)



            # for i in range(self.opt.loadSize-20, 0, -1):
            if np.sum(birdy['pure_bg_area']) / np.sum(np.ones(np.shape(birdy['pure_bg_area']))) < 0.3:
                # print('continue')
                continue
            # else:
            #     print('able to composite')
            # for i in range(self.opt.loadSize-40):
            # for i in range(self.opt.loadSize-40, 40, -10):
            pixel_th = int(self.opt.loadSize * (20/256))
            pixel_bottom = int(self.opt.loadSize * (20/256))
            interval = int(self.opt.loadSize * (40/256))

            counts = 0
            # records = []
            for i in range(self.opt.loadSize-1*pixel_th, 1*pixel_th, -interval):
                for j in range(self.opt.loadSize-1*pixel_th, 1*pixel_th, -interval):
                    birdy_new = birdy.copy()
                    # print('offset',i)
                # for i in range(2*pixel_th,self.opt.loadSize-2*pixel_th, interval):
                # for i in range(40, self.opt.loadSize-40, 30):
                #     x_offset = int( (self.opt.loadSize - col_length - 1)/self.opt.loadSize * (i + pixel_th) )
                #     y_offset = int( (self.opt.loadSize - row_length - 1)/self.opt.loadSize * (i + pixel_bottom) )
                    x_offset = i
                    y_offset = j

                    new_foreground_object = np.zeros(np.shape(birdy['A']))
                    new_foreground_object_mask = np.zeros(np.shape(birdy['bg_instance']))
                    new_foreground_shadow_mask = np.zeros(np.shape(birdy['bg_instance']))

                    t1 = new_foreground_object_mask[x_offset:x_offset + col_length, y_offset:y_offset+row_length]
                    t11 = fg_instance[min(x):max(x), min(y):max(y)]
                    t2 = new_foreground_shadow_mask[x_offset:x_offset + col_length, y_offset:y_offset+row_length]
                    t22 = fg_shadow[min(x):max(x), min(y):max(y)]
                    if np.shape(t1)!=np.shape(t11) or np.shape(t2)!=np.shape(t22):
                        continue

                    new_foreground_object_mask[x_offset:x_offset + col_length, y_offset:y_offset+row_length] = fg_instance[min(x):max(x), min(y):max(y)]
                    new_foreground_shadow_mask[x_offset:x_offset + col_length, y_offset:y_offset+row_length] = fg_shadow[min(x):max(x), min(y):max(y)]
                    new_foreground_object_mask[new_foreground_object_mask>0] = 255
                    new_foreground_shadow_mask[new_foreground_shadow_mask>0] = 255

                    new_foreground_object[x_offset:x_offset + col_length, y_offset:y_offset+row_length ,:] = A_img_foreground[min(x):max(x), min(y):max(y),:]

                    overlap = (new_foreground_object_mask) * birdy['pure_bg_area']


                    if np.sum(overlap) == np.sum(new_foreground_object_mask):
                        counts+=1

                        birdy_new['instancemask'] = Image.fromarray(np.uint8(new_foreground_object_mask), mode='L')
                        # records.append(new_foreground_object_mask)
                        # fg_shadow = Image.fromarray(np.uint8(new_foreground_shadow_mask), mode='L')
                        birdy_new['B'] = Image.fromarray(np.uint8(new_foreground_shadow_mask), mode='L')
                        ####replacing the area of foreground
                        final_img = new_foreground_object * (np.tile(np.expand_dims(new_foreground_object_mask / 255, -1), (1, 1, 3))) + \
                                    birdy['A'] * (1 - np.tile(np.expand_dims(np.array(new_foreground_object_mask) / 255, -1), (1, 1, 3)))
                        birdy_new['C'] = Image.fromarray(np.uint8(final_img), mode='RGB')
                        list_birdy.append(birdy_new)
                        # del birdy['instancemask'], birdy['B'],birdy['C']



                # print('again')
                        # break

                    else:
                        continue
            print('placement num', counts)

        new_birdy_list = []
        # for birdy in list_birdy:
        for i in range(len(list_birdy)):
            current_dict = {}
            for k,im in list_birdy[i].items():
                if  k=='ratio' or k == 'max_object':
                    current_dict[k] = list_birdy[i][k]
                    continue
                current_dict[k] = self.transformB(im)

            for k,im in current_dict.items():
                if  k=='ratio' or k == 'max_object':
                    continue
                current_dict[k] = (im - 0.5)*2


            for k,im in current_dict.items():
                if  k=='ratio' or k == 'max_object':
                    continue
                im = F.interpolate(im.unsqueeze(0), size = self.opt.loadSize)
                current_dict[k] = im.squeeze(0)

            current_dict['w'] = ow
            current_dict['h'] = oh
            new_birdy_list.append(current_dict)

        return new_birdy_list

    def __len__(self):
        return self.data_size

    def name(self):
        return 'CompositeDataset'
