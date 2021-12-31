#!/usr/bin/env python
# coding: utf-8

# In[3]:

# from shapedetector import ShapeDetector
# from colorlabeler import ColorLabeler
# import argparse
# import imutils


import os
from os import listdir
from os.path import isfile, join
from PIL import Image as Image
import numpy as np
import pandas as pd


def obtain_tuple(data_root, folders, foreground_mask_folder):
    store_path_paired_image = os.path.join(data_root, 'BTscore_image_comparison')
    store_path_list = os.path.join(data_root, 'BTscore_image_comparison_list.xlsx')
    if not os.path.exists(store_path_paired_image):
        os.makedirs(store_path_paired_image)

    combined_images_list = []

    for root, dirs, files in os.walk(os.path.join(data_root ,foreground_mask_folder[0]) ):
        if (len(dirs) < 1):
            print(len(dirs))
            print('root', root)
            # print('dir', dirs)
            unit_size = 256
            target_width = 256*3
            new_image = Image.new('RGB', (target_width, unit_size))
            print(root.split('/')[-1])

            root_index = root.split('/')[-1].split('s')[-1]
            print('root_index',root_index)
            for file in files:
                current_foreground_mask_path = os.path.join(root, file)
                fg_mask_image = Image.open(current_foreground_mask_path)
                new_image.paste(fg_mask_image, (256, 0, 256*2, unit_size))
                file_index = file.split('_')[0].split('t')[-1]
                print('image index', file_index)
                for i in range(len(folders)):
                    folder1 = folders[i]
                    image1_root = root.replace(foreground_mask_folder[0], folder1)
                    image1_root_filelist = [f for f in listdir(image1_root) if isfile(join(image1_root, f)) and f.endswith('png')]
                    ###matching image
                    for image1_name in image1_root_filelist:
                        if image1_name.split('_')[0] == file_index:
                            image1_path = os.path.join(image1_root, image1_name)
                            print('image1 path', image1_path)
                            image1 = Image.open(image1_path)
                            new_image.paste(image1, (0, 0, 256, unit_size))
                    for j in range(i+1, len(folders)):
                        folder2 = folders[j]
                        image2_root = root.replace(foreground_mask_folder[0], folder2)
                        image2_root_filelist = [f for f in listdir(image2_root) if isfile(join(image2_root, f)) and f.endswith('png')]
                        ###matching image
                        for image2_name in image2_root_filelist:
                            if image2_name.split('_')[0] == file_index:
                                image2_path = os.path.join(image2_root, image2_name)
                                print('image2 path', image2_path)
                                image2 = Image.open(image2_path)
                                new_image.paste(image2, (256*2, 0, 256*3, unit_size))
                        #####complete
                        new_image_name = 'Images{}_{}-Methods{}_{}.png'.format(root_index,file_index,i,j)
                        new_image.save(os.path.join(store_path_paired_image, new_image_name))
                        combined_images_list.append([new_image_name, []])


    name=['image_name','which side is more realistic (1 means the left one, 2 means the right one)']
    output=pd.DataFrame(columns=name,data=combined_images_list)#数据有三列，列名分别为one,two,three
    # output.sort_values('image_name')
    output.to_excel(store_path_list, index=False)

    # output_excel = {'image_name':[], 'which side is more realistic (1 means the left one, 2 means the right one)':[]}
    # output_excel['image_name'] = combined_images_list
    # output_excel['which side is more realistic (1 means the left one, 2 means the right one'] = []
    # output = pd.DataFrame(output_excel)
    # output.to_excel(store_path_list, index=False)





    # for i,folder in enumerate (folders):
    #     current_dataset = os.path.join(data_root, folder)
    #     im_list  =  [f for f in listdir(current_dataset) if isfile(join(current_dataset, f)) and f.endswith('jpg')]
    #     for im in im_list[:]:
    #         current_shadowimg_path = os.path.join(current_dataset,im)
    #         current_shadowimg = cv2.imread(current_shadowimg_path)
    #         store_shadowimg_path = os.path.join(store_path_shadowimage,im)
    #         # cv2.imwrite(store_shadowimg_path, current_shadowimg)

root = '/media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/ShadowGeneration/HYShadowGeneration/VISUALIZATION/BTscores/'
folders = [ 'pix2pix', 'pix2pixresidual', 'shadowgan', 'maskshadowgan', 'arshadowgan', 'ours']
foreground_mask_folder = ['foregroundmask']

obtain_tuple(root, folders, foreground_mask_folder)