import argparse, os
from PIL import Image
from data import CreateDataLoader
import  util
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',default='/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/SOBAFinalSplit/',
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    parser.add_argument('--dataset_mode', type=str, default='DesobaSyntheticImageGeneration', help='chooses how datasets are loaded. [unaligned | aligned | single]')
    parser.add_argument('--batch_size', type=int, default=1, help='scale images to this size')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--bosfree', action='store_true')
    parser.add_argument('--bos', action='store_true')

    

    parser.add_argument('--output_path', default='/media/user/data/ShadowGeneration/HYShadowGeneration/SGRNet/DESOBA_DATASET/TestSplit/',type=str)

    return parser


if __name__ == "__main__":
    # Visualizing examples
    parser = get_parser()
    opt, _ = parser.parse_known_args()
    opt.isTrain = False
    opt.shadowimg_path = opt.dataroot + 'shadowimg'
    opt.shadowfree_path = opt.dataroot + 'shadowfree'
    opt.instance_path = opt.dataroot + 'shadowcolorinstancemask'
    opt.shadow_path = opt.dataroot + 'shadowcolormask'
    opt.bg_instance_path = opt.dataroot + 'shadowcolorinstancemask'
    opt.bg_shadow_path = opt.dataroot + 'shadowcolormask'
    opt.new_mask_path = opt.dataroot + 'shadownewmask'
    opt.param_path = opt.dataroot + 'SOBA_params'


    if opt.bos:
        opt.output_path = opt.output_path + 'bos/'
    elif opt.bosfree:
        opt.output_path = opt.output_path + 'bosfree/'


    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    loadSize = opt.loadSize

    for i, data in enumerate(dataset):
        synthetic_composite_img = util.tensor2im(data['Synth_img'])
        shadow_img = util.tensor2im(data['Shadow_img'])
        fg_instance_mask = util.tensor2im(data['fg_instance_mask'])
        fg_shadow_mask = util.tensor2im(data['fg_shadow_mask'])
        bg_instance_mask = util.tensor2im(data['bg_instance_mask'])
        bg_shadow_mask = util.tensor2im(data['bg_shadow_mask'])

        synthetic_path = opt.output_path + 'shadowfree_img'
        gt_path =  opt.output_path + 'shadow_img'
        fginstance_path = opt.output_path + 'foreground_object_mask'
        fgshadow_path = opt.output_path + 'foreground_shadow_mask'
        bginstance_path = opt.output_path + 'background_object_mask'
        bgshadow_path = opt.output_path + 'background_shadow_mask'

        paths = [synthetic_path, gt_path, fginstance_path, fgshadow_path, bginstance_path, bgshadow_path]
        imgs = [synthetic_composite_img,shadow_img, fg_instance_mask, fg_shadow_mask, bg_instance_mask,bg_shadow_mask]

        for j, path in enumerate(paths):
            if not os.path.exists(path):
                os.makedirs(path)
            output_name = '{}.png'.format(i)
            save_path = '%s/%s' % (path, output_name)
            util.save_image(imgs[j], save_path)
