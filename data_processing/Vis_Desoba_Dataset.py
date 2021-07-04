import argparse, os
from PIL import Image
from data import CreateDataLoader
import  util
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',default='/media/user/data/ShadowGeneration/InstanceShadowDetection/SOBA/SOBAFinalSplit/',
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--shadowimg_path', default='/media/user/data/ShadowGeneration/InstanceShadowDetection/DESOBA/ShadowImage',type=str)
    parser.add_argument('--shadowfree_path', default='/media/user/data/ShadowGeneration/InstanceShadowDetection/DESOBA/DeshadowedImage',type=str)
    parser.add_argument('--instance_path', default='/media/user/data/ShadowGeneration/InstanceShadowDetection/DESOBA/InstanceMask',type=str)
    parser.add_argument('--shadow_path', default='/media/user/data/ShadowGeneration/InstanceShadowDetection/DESOBA/ShadowMask',type=str)
    parser.add_argument('--output_path', default='/media/user/data/ShadowGeneration/InstanceShadowDetection/DESOBA/Visualization_Examples',type=str)
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    parser.add_argument('--dataset_mode', type=str, default='DesobaSyntheticImageGeneration')
    parser.add_argument('--batch_size', type=int, default=1, help='scale images to this size')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

    return parser


if __name__ == "__main__":
    # Visualizing examples
    parser = get_parser()
    opt, _ = parser.parse_known_args()
    opt.isTrain = False
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    loadSize = opt.loadSize

    for i, data in enumerate(dataset):
        row = []
        new_img = Image.new('RGB', (loadSize*6, loadSize), 255)
        synthetic_composite_img = util.tensor2im(data['Synth_img'])
        shadow_img = util.tensor2im(data['Shadow_img'])
        fg_instance_mask = util.tensor2im(data['fg_instance_mask'])
        fg_shadow_mask = util.tensor2im(data['fg_shadow_mask'])
        bg_instance_mask = util.tensor2im(data['bg_instance_mask'])
        bg_shadow_mask = util.tensor2im(data['bg_shadow_mask'])

        row.append(synthetic_composite_img)
        row.append(shadow_img)
        row.append(fg_instance_mask)
        row.append(fg_shadow_mask)
        row.append(bg_instance_mask)
        row.append(bg_shadow_mask)
        row = tuple(row)
        row = np.hstack(row)

        if not os.path.exists(opt.output_path):
            os.makedirs(opt.output_path)
        output_name = '{}.png'.format(i)
        save_path = '%s/%s' % (opt.output_path, output_name)
        util.save_image(row, save_path)
