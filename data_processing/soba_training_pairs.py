import cv2
import os
from os import listdir
from os.path import isfile, join
from PIL import Image as Image
import numpy as np

from scipy.optimize import curve_fit
import math
import json,time


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# load json data from file
def load(dataroot):
    with open(dataroot,encoding='utf-8-sig', errors='ignore') as f:
        data = json.load(f, strict=False)
        # line = f.readline()
        return data
        # f.close()
        # return line

def SOBA_annotation_data(data_root, store_root):
    train_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/annotations/SOBA_train.json'
    test_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/annotations/SOBA_val.json'
    train_data = load(train_root)
    test_data = load(test_root)

    colormask_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/NewTemp/SOBAShadowMask/'
    colorinstance_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/NewTemp/SOBAInstanceMask/'

    store_path_shadowmask = store_root + 'shadowmask/'
    store_path_newshadowmask = store_root + 'shadownewmask/'
    store_path_shadowimg = store_root + 'shadowimg/'
    store_path_shadowfree = store_root + 'shadowfree/'
    store_path_shadowcompare = store_root + 'shadowcompare/'
    store_path_colormask = store_root + 'shadowcolormask/'
    store_path_colorinstancemask = store_root + 'shadowcolorinstancemask/'

    if not os.path.exists(store_path_newshadowmask):
        os.makedirs(store_path_newshadowmask)
    if not os.path.exists(store_path_shadowimg):
        os.makedirs(store_path_shadowimg)
    if not os.path.exists(store_path_shadowmask):
        os.makedirs(store_path_shadowmask)
    if not os.path.exists(store_path_shadowfree):
        os.makedirs(store_path_shadowfree)
    if not os.path.exists(store_path_shadowcompare):
        os.makedirs(store_path_shadowcompare)
    if not os.path.exists(store_path_colormask):
        os.makedirs(store_path_colormask)
    if not os.path.exists(store_path_colorinstancemask):
        os.makedirs(store_path_colorinstancemask)



    train_image_path_list = []
    train_image_names = []
    for image in train_data['images']:
        train_image_path_list.append(os.path.join('/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/SOBA/',image['file_name'])  )
        train_image_names.append(image['file_name'].split('/')[-1])

    test_image_path_list = []
    test_image_names = []
    for image in test_data['images']:
        test_image_path_list.append(os.path.join('/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/SOBA/',image['file_name'])  )
        test_image_names.append(image['file_name'].split('/')[-1])

    total_dirs = []
    circle_dirs = []
    mask_dirs = []
    shadow_dirs = []
    shadow_free_dirs = []
    for root,dirs,files in os.walk(data_root):
        for file in files:
            total_dirs.append(file)
            file_name = os.path.join(root, file)
            # print(file_name)
            # if file_name.split('.')[-1] == 'jpg':
            if len(file_name.split('-mask')) == 2:
                mask_dirs.append(file_name)
                # shadow_dirs.append(file_name.split('-mask')[0] + '-Shadow.png')
            elif len(file_name.split('-Shadow')) == 2 and ('left' in file_name.split('/')):
                if os.path.exists(file_name):
                    shadow_dirs.append(file_name)
                else:
                    print('no shadow image {}'.format(file))
                    raise
                # print('hhhhhhh',file_name.split('-Shadow'))
            elif len(file_name.split('-Circled')) == 2:
                circle_dirs.append(file_name)

            # elif ((len(file_name.split('-mask')) != 2) and (len(file_name.split('-Shadow')) != 2) and (len(file_name.split('-Circled')) != 2)):
            elif ('right' in file_name.split('/')):
                if os.path.exists(file_name):
                    shadow_free_dirs.append(file_name)
                else:
                    print('no shadowfree image {}'.format(file))
                    raise

    print('0',len(total_dirs))
    print('1', len(mask_dirs))
    print('2',len(shadow_dirs))
    print('3',len(circle_dirs))
    print('4',len(shadow_free_dirs))

    ########## storing {mask, shadow, shadowfree} #########
    #####storing shadow mask
    if len(os.listdir(store_path_shadowmask) ) != len(mask_dirs):
        mask_names = []
        for file in mask_dirs:
            # print('mask',file)
            mask = cv2.imread(file)
            new_name =  store_path_shadowmask + file.split('/')[-1].split('-mask')[0] + '.png'
            mask_names.append(file.split('/')[-1].split('-mask')[0])
            if not os.path.exists(new_name):
                cv2.imwrite(new_name, mask)
        print('mask length', len(mask_names))
    else:
        mask_names = [(file.split('/')[-1].split('.')[0]) for file in (os.listdir(store_path_shadowmask))]

    ######storing shadow image
    if len(os.listdir(store_path_shadowimg)) != len(shadow_dirs):
        shadow_names = []
        for file in shadow_dirs:
            print('shadowimg',file)
            shadowimg = cv2.imread(file)
            new_name =  store_path_shadowimg + file.split('/')[-1].split('-Shadow')[0] + '.png'
            if file.split('/')[-1].split('-Shadow')[0] not in mask_names:
                print('shadowimg not match mask', file.split('/')[-1].split('-Shadow')[0])
                raise
            shadow_names.append(file.split('/')[-1].split('-Shadow')[0])
            if not os.path.exists(new_name):
                cv2.imwrite(new_name, shadowimg )
        print('shadowimg length', len(shadow_names))

    ######storing shadowfree image
    if len(os.listdir(store_path_shadowfree)) != len(shadow_free_dirs):
        shadowfree_names = []
        for file in shadow_free_dirs:
            print('shadowfree',file)
            shadowfree = cv2.imread(file)
            new_name =  store_path_shadowfree + file.split('/')[-1]
            if file.split('/')[-1].split('.')[0] not in mask_names:
                print('shadowfree not match mask', file.split('/')[-1])
                raise
            shadowfree_names.append(file.split('/')[-1].split('.')[0])
            if not os.path.exists(new_name):
                cv2.imwrite(new_name, shadowfree)
        print('shadowfree length', len(shadowfree_names))

    ########## storing {mask, shadow, shadowfree} #########


    ####calculating new mask
    is_errors = False
    if is_errors:
        mask_errors = []
        image_errors = []
        imgs = os.listdir(store_path_shadowmask)
        for im in imgs:
            index = im.split('/')[-1]
            # print('hhhh', store_path_shadowimg + index)
            # print('gggg', store_path_shadowfree + index)
            if os.path.exists(store_path_shadowimg + index):
                shadow_img = cv2.imread(store_path_shadowimg + index).astype(np.float32)
            if os.path.exists(store_path_shadowfree + index):
                shadowfree = cv2.imread(store_path_shadowfree + index).astype(np.float32)
            if os.path.exists(store_path_shadowmask + index):
                shadowmask = cv2.imread(store_path_shadowmask + index).astype(np.float32)
                shadowmask = cv2.cvtColor(shadowmask,cv2.COLOR_BGR2GRAY)
                _, shadowmask = cv2.threshold(shadowmask,0,255,cv2.THRESH_BINARY)
                # current_shadowmask = gray
                shadowmask_norm =  np.expand_dims(shadowmask / 255,2)

            if np.shape(shadowfree) == np.shape(shadow_img):
                new_mask = shadowfree - shadow_img
                # print('mean',np.mean(new_mask))
                new_mask = cv2.cvtColor(new_mask,cv2.COLOR_BGR2GRAY)
                _, new_mask = cv2.threshold(new_mask,0,255,cv2.THRESH_BINARY)
                error_mask = (np.abs(shadowmask - new_mask)).mean()
                mask_errors.append(error_mask)
                print('mask error', error_mask)

                img_error = np.mean( np.abs(shadow_img - shadowfree) *  ( 1 - shadowmask_norm) )
                print('image error', img_error)
                image_errors.append(img_error)


                compare_mask = np.concatenate([new_mask, shadowmask],axis=1)
                compare_mask = compare_mask.astype(np.uint8)
                new_name = store_path_shadowcompare + index
                if not os.path.exists(new_name):
                    cv2.imwrite(new_name, compare_mask)

                new_mask = new_mask.astype(np.uint8)
                new_name = store_path_newshadowmask + index
                if not os.path.exists(new_name):
                    cv2.imwrite(new_name, new_mask)
        print('mean mask error', np.mean(np.array(mask_errors)))
        print('mean image error', np.mean(np.array(image_errors)))



    #####storing color shadow_mask/instance_mask
    imgs = os.listdir(store_path_shadowcompare)
    for im in imgs:
        index = im.split('/')[-1].split('.')[0]
        color_shadowmask = colormask_root + index + '-3.png'
        color_instancemask = colorinstance_root  + index + '-2.png'
        if os.path.exists(color_instancemask) and os.path.exists(color_shadowmask):
            colorshadow = cv2.imread(color_shadowmask)
            colorinstance = cv2.imread(color_instancemask)
        else:
            print('not match', index)
        if not os.path.exists(store_path_colormask + im):
            cv2.imwrite(store_path_colormask + im, colorshadow)
        if not os.path.exists(store_path_colorinstancemask + im):
            cv2.imwrite(store_path_colorinstancemask + im, colorinstance)



def SOBA_annotation_data_new(data_root, store_root):
    train_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/annotations/SOBA_train.json'
    test_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/annotations/SOBA_val.json'
    train_data = load(train_root)
    test_data = load(test_root)

    colormask_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/OriginalSplite/SOBAShadowMask/'
    colorinstance_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/OriginalSplite/SOBAInstanceMask/'

    store_path_shadowmask = store_root + 'shadowmask/'
    store_path_newshadowmask = store_root + 'shadownewmask/'
    store_path_shadowimg = store_root + 'shadowimg/'
    store_path_shadowfree = store_root + 'shadowfree/'
    store_path_shadowcompare = store_root + 'shadowcompare/'
    store_path_colormask = store_root + 'shadowcolormask/'
    store_path_colorinstancemask = store_root + 'shadowcolorinstancemask/'

    if not os.path.exists(store_path_newshadowmask):
        os.makedirs(store_path_newshadowmask)
    if not os.path.exists(store_path_shadowimg):
        os.makedirs(store_path_shadowimg)
    if not os.path.exists(store_path_shadowmask):
        os.makedirs(store_path_shadowmask)
    if not os.path.exists(store_path_shadowfree):
        os.makedirs(store_path_shadowfree)
    if not os.path.exists(store_path_shadowcompare):
        os.makedirs(store_path_shadowcompare)
    if not os.path.exists(store_path_colormask):
        os.makedirs(store_path_colormask)
    if not os.path.exists(store_path_colorinstancemask):
        os.makedirs(store_path_colorinstancemask)



    train_image_path_list = []
    train_image_names = []
    for image in train_data['images']:
        train_image_path_list.append(os.path.join('/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/SOBA/',image['file_name'])  )
        train_image_names.append(image['file_name'].split('/')[-1])
        # print('training', image['file_name'].split('/')[-1])

    test_image_path_list = []
    test_image_names = []
    for image in test_data['images']:
        test_image_path_list.append(os.path.join('/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/SOBA/',image['file_name'])  )
        test_image_names.append(image['file_name'].split('/')[-1])
        # print('testing', image['file_name'].split('/')[-1])


    print('original training and testing data length',len(train_image_names), len(test_image_names))

    total_dirs = []
    circle_dirs = []
    mask_dirs = []
    shadow_dirs = []
    shadow_free_dirs = []
    for root,dirs,files in os.walk(data_root):
        for file in files:
            total_dirs.append(file)
            file_name = os.path.join(root, file)
            if len(file_name.split('-mask')) == 2:
                mask_dirs.append(file_name)
                index = file_name.split('/')[-1].split('-mask')[0]
                mask = cv2.imread(file_name)
                new_name_mask =  store_path_shadowmask + index + '.png'
                if not os.path.exists(new_name_mask):
                    cv2.imwrite(new_name_mask, mask)

                shadow_path = os.path.join(root, index+'-Shadow.png')
                if not os.path.exists(shadow_path):
                    print('no corresponding shadow', shadow_path)
                else:
                    shadow = cv2.imread(shadow_path)
                    new_name_shadow =  store_path_shadowimg + index + '.png'
                    if not os.path.exists(new_name_shadow):
                        cv2.imwrite(new_name_shadow, shadow)

                shadowfree_path1 = os.path.join(root, index+'.png')
                shadowfree_path2 = os.path.join(root, index+'.jpg')
                if not os.path.exists(shadowfree_path1):
                    print('no corresponding shadowfree png', shadowfree_path1)
                    if not os.path.exists(shadowfree_path2):
                        print('no corresponding shadowfree jpg', shadowfree_path2)
                    # else:
                    #     shadowfree = cv2.imread(shadowfree_path2)
                    #     new_name_shadowfree =  store_path_shadowfree + index + '.png'
                    #     if not os.path.exists(new_name_shadowfree):
                    #         cv2.imwrite(new_name_shadowfree, shadowfree)
                else:
                    shadowfree = cv2.imread(shadowfree_path1)
                    new_name_shadowfree =  store_path_shadowfree + index + '.png'
                    if not os.path.exists(new_name_shadowfree):
                        cv2.imwrite(new_name_shadowfree, shadowfree)


    print('0',len(total_dirs))
    print('1', len(mask_dirs))
    print('2',len(shadow_dirs))
    print('3',len(circle_dirs))
    print('4',len(shadow_free_dirs))

    ########## storing {mask, shadow, shadowfree} #########


    ######create new training and testing split
    imgs = os.listdir(store_path_shadowfree)
    print('valid data length',len(imgs))
    # print('all new images', imgs)


    for im in train_image_names:
        if im not in imgs:
            if (im.split('.')[0]+'.png') not in imgs:
                print('trianing not include', im)
                train_image_names.remove(im)
        else:
            im.replace('.jpg','.png')
    for im in test_image_names:
        if im not in imgs:
            if (im.split('.')[0]+'.png') not in imgs:
                print('testing not include', im)
                test_image_names.remove(im)
        else:
            im.replace('.jpg','.png')

    # print(train_image_names, test_image_names)
    New_training_labels = []
    New_testing_labels = []
    New_remaning_labels = []
    for im in train_image_names:
        New_training_labels.append(im.split('.')[0]+'.png')
    for im in test_image_names:
        New_testing_labels.append(im.split('.')[0]+'.png')

    for im in imgs:
        if im not in New_training_labels:
            if im not in New_testing_labels:
                New_remaning_labels.append(im)





    print('new training and testing split, remaining', len(New_training_labels),len(New_testing_labels), len(New_remaning_labels))
    #####storing txt
    Training_txt = open(store_root + 'Training_labels.txt', 'w')
    Testing_txt = open(store_root + 'Testing_labels.txt', 'w')
    Remaining_txt = open(store_root + 'Remaining_labels.txt', 'w')
    for tra in New_training_labels:
        Training_txt.write(tra)
        Training_txt.write('\n')
    Training_txt.close()

    for tes in New_testing_labels:
        Testing_txt.write(tes)
        Testing_txt.write('\n')
    Testing_txt.close()


    for re in New_remaning_labels:
        Remaining_txt.write(re)
        Remaining_txt.write('\n')
    Remaining_txt.close()



    # print('new training', New_training_labels)
    # print('new testing', New_testing_labels)
    # print('new remaining', New_remaning_labels)





    ####calculating new mask
    is_errors = True
    if is_errors:
        mask_errors = []
        image_errors = []
        imgs = os.listdir(store_path_shadowfree)
        for im in imgs:
            index = im.split('/')[-1]
            # print('hhhh', store_path_shadowimg + index)
            # print('gggg', store_path_shadowfree + index)
            if os.path.exists(store_path_shadowimg + index):
                shadow_img = cv2.imread(store_path_shadowimg + index).astype(np.float32)
            if os.path.exists(store_path_shadowfree + index):
                shadowfree = cv2.imread(store_path_shadowfree + index).astype(np.float32)
            if os.path.exists(store_path_shadowmask + index):
                shadowmask = cv2.imread(store_path_shadowmask + index).astype(np.float32)
                shadowmask = cv2.cvtColor(shadowmask,cv2.COLOR_BGR2GRAY)
                _, shadowmask = cv2.threshold(shadowmask,0,255,cv2.THRESH_BINARY)
                # current_shadowmask = gray
                shadowmask_norm =  np.expand_dims(shadowmask / 255,2)

            if np.shape(shadowfree) == np.shape(shadow_img):
                new_mask = shadowfree - shadow_img
                # print('mean',np.mean(new_mask))
                new_mask = cv2.cvtColor(new_mask,cv2.COLOR_BGR2GRAY)
                _, new_mask = cv2.threshold(new_mask,0,255,cv2.THRESH_BINARY)
                if np.shape(shadowmask) == np.shape(new_mask):
                    img_error = np.mean( np.abs(shadow_img - shadowfree) *  ( 1 - shadowmask_norm) )
                    # print('image error', img_error)
                    image_errors.append(img_error)

                    error_mask = (np.abs(shadowmask - new_mask)).mean()
                    mask_errors.append(error_mask)
                    # print('mask error', error_mask)
                    compare_mask = np.concatenate([new_mask, shadowmask],axis=1)
                    compare_mask = compare_mask.astype(np.uint8)
                    new_name = store_path_shadowcompare + index
                    if not os.path.exists(new_name):
                        cv2.imwrite(new_name, compare_mask)

                    new_mask = new_mask.astype(np.uint8)
                    new_name = store_path_newshadowmask + index
                    if not os.path.exists(new_name):
                        cv2.imwrite(new_name, new_mask)

                else:
                    print('mask shape not match',im)
            else:
                print('shadow shape not match shadowfree',im)





        print('mean mask error', np.mean(np.array(mask_errors)))
        print('mean image error', np.mean(np.array(image_errors)))



    #####storing color shadow_mask/instance_mask
    imgs = os.listdir(store_path_shadowcompare)
    for im in imgs:
        index = im.split('/')[-1].split('.')[0]
        color_shadowmask = colormask_root + index + '.png'
        color_instancemask = colorinstance_root  + index + '.png'
        if os.path.exists(color_instancemask) and os.path.exists(color_shadowmask):
            colorshadow = cv2.imread(color_shadowmask)
            colorinstance = cv2.imread(color_instancemask)
            if not os.path.exists(store_path_colormask + im):
                cv2.imwrite(store_path_colormask + im, colorshadow)
            if not os.path.exists(store_path_colorinstancemask + im):
                cv2.imwrite(store_path_colorinstancemask + im, colorinstance)
        else:
            print('color instance-shadow mask not match', index)







# data_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/SOBAFinal/'
# store_root = '/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/SOBAFinalSplit/'
# SOBA_annotation_data_new(data_root, store_root)