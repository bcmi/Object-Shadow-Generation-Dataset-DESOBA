import argparse, os
from PIL import Image
from data import CreateDataLoader
import util
import numpy as np
import torch

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',default='../DESOBA_DATASET/',
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--output_path', default='../DESOBA_DATASET/CompositeImages/',type=str)

    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    # parser.add_argument('--dataset_mode', type=str, default='TwoForegroundObjectsComposite', help='chooses how datasets are loaded. [unaligned | aligned | single]')
    parser.add_argument('--dataset_mode', type=str, default='Composite', help='chooses how datasets are loaded. [unaligned | aligned | single]')

    parser.add_argument('--batch_size', type=int, default=1, help='scale images to this size')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--isTrain', type=int, default=0)
    parser.add_argument('--foreground_object_num', type=int, default=1)

    return parser


if __name__ == "__main__":
    # Visualizing examples
    parser = get_parser()
    opt, _ = parser.parse_known_args()
    opt.shadowimg_path = opt.dataroot + '/ShadowImage'
    opt.shadowfree_path = opt.dataroot + '/DeshadowedImage'
    opt.instance_path = opt.dataroot + '/InstanceMask'
    opt.shadow_path = opt.dataroot + '/ShadowMask'
    opt.new_mask_path = opt.dataroot + '/shadownewmask'

    opt.store_path = opt.output_path + '/{}_ForegroundObject/'.format(opt.foreground_object_num)


    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    loadSize = opt.loadSize
    length = 0

    for i, data_list in enumerate(dataset): 
        for k, data in enumerate(data_list):
            synthetic_composite_img = util.tensor2im(data['C'])
            shadow_img = util.tensor2im(data['A'])
            fg_instance_mask = util.tensor2im(data['instancemask'])
            fg_shadow_mask = util.tensor2im(data['B'])
            bg_instance_mask = util.tensor2im(data['bg_instance'])
            bg_shadow_mask = util.tensor2im(data['bg_shadow'])

            synthetic_path = opt.store_path + 'shadowfree'
            gt_path =  opt.store_path + 'shadowimg'
            fginstance_path = opt.store_path + 'foreground_object_mask'
            fgshadow_path = opt.store_path + 'foreground_shadow_mask'
            bginstance_path = opt.store_path + 'background_object_mask'
            bgshadow_path = opt.store_path + 'background_shadow_mask'

            paths = [synthetic_path, gt_path, fginstance_path, fgshadow_path, bginstance_path, bgshadow_path]
            imgs = [synthetic_composite_img,shadow_img, fg_instance_mask, fg_shadow_mask, bg_instance_mask,bg_shadow_mask]

            for j, path in enumerate(paths):
                if not os.path.exists(path):
                    os.makedirs(path)
                output_name = '{}.png'.format(k+length)
                save_path = '%s/%s' % (path, output_name)
                util.save_image(imgs[j], save_path)

        length +=len(data_list)
        print('producing {} images'.format(length))
