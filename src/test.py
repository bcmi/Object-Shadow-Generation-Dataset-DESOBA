from collections import OrderedDict
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from PIL import Image
import visdom
from util.util import sdmkdir
import time
import tqdm
import numpy as np
import math
import torch

opt = TestOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = create_model(opt)
model.setup(opt)
visualizer = Visualizer(opt)


RMSES = []
shadowRMSES = []
SSIMS = []
shadowSSIMS = []


for i, data in enumerate(dataset):
    model.set_input(data)
    model.prediction()
    lenght, visual_dict = model.get_current_visuals()
    visualizer.display_current_results(visual_dict, i)
    RMSE,shadowRMSE, SSIM, shadowSSIM = model.get_current_errors()
    RMSES.append(RMSE)
    shadowRMSES.append(shadowRMSE)
    SSIMS.append(SSIM)
    shadowSSIMS.append(shadowSSIM)


RMSES_final =  np.mean(np.array(RMSES))
shadowRMSES_final =  np.mean(np.array(shadowRMSES))
SSIMS_final =  (torch.mean(torch.tensor(SSIMS)))
shadowSSIMS_final = torch.mean(torch.tensor(shadowSSIMS))

if opt.bos:
	print('totally {} test bos images'.format(len(RMSES)))
elif opt.bosfree:
	print('totally {} test bosfree images'.format(len(RMSES)))

print('final rmse is', RMSES_final)
print('final shadowrmse is', shadowRMSES_final)
print('final ssim is', SSIMS_final)
print('final shadowssim is', shadowSSIMS_final)

   