import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
# from visdom import Visdom
import torchvision.transforms as transforms
import numpy as np
from skimage.filters import threshold_otsu


to_pil = transforms.ToPILImage()
to_gray = transforms.Grayscale(num_output_channels=1)

class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1]



def mask_generator(shadow, shadow_free):
    im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu()))
    im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu()))

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow
    mask.requires_grad = False

    return mask



def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

# class Logger():
#     def __init__(self, n_epochs, batches_epoch, server='http://137.189.90.150', http_proxy_host='http://proxy.cse.cuhk.edu.hk/', env = 'main'):
#         self.viz = Visdom(server = server, http_proxy_host = http_proxy_host, env = env)#, http_proxy_port='http://proxy.cse.cuhk.edu.hk:8000/')
#         self.n_epochs = n_epochs
#         self.batches_epoch = batches_epoch
#         self.epoch = 1
#         self.batch = 1
#         self.prev_time = time.time()
#         self.mean_period = 0
#         self.losses = {}
#         self.loss_windows = {}
#         self.image_windows = {}
#
#
#     def log(self, losses=None, images=None):
#         self.mean_period += (time.time() - self.prev_time)
#         self.prev_time = time.time()
#
#         sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
#
#         for i, loss_name in enumerate(losses.keys()):
#             if loss_name not in self.losses:
#                 self.losses[loss_name] = losses[loss_name].data.item()
#             else:
#                 self.losses[loss_name] += losses[loss_name].data.item()
#
#             if (i+1) == len(losses.keys()):
#                 sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
#             else:
#                 sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))
#
#         batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
#         batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
#         sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))
#
#         # Draw images
#         for image_name, tensor in images.items():
#             if image_name not in self.image_windows:
#                 self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
#             else:
#                 self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})
#
#         # End of epoch
#         if (self.batch % self.batches_epoch) == 0:
#             # Plot losses
#             for loss_name, loss in self.losses.items():
#                 if loss_name not in self.loss_windows:
#                     self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
#                                                                     opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
#                 else:
#                     self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
#                 # Reset losses for next epoch
#                 self.losses[loss_name] = 0.0
#
#             self.epoch += 1
#             self.batch = 1
#             sys.stdout.write('\n')
#         else:
#             self.batch += 1
#
#

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



