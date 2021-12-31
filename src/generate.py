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


lengths = 0
for i, data in enumerate(dataset):
    model.set_input(data)
    model.prediction()
    length, visual_dict = model.get_current_visuals()
    visualizer.display_current_results(visual_dict, i)
    lengths+=length

if opt.bos:
	print('totally {} test bos images'.format(lengths))
elif opt.bosfree:
	print('totally {} test bosfree images'.format(lengths))
else:
    print('totally {} real composite images'.format(lengths))


