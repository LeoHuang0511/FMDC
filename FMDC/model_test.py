import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from config import cfg
from importlib import import_module
from model.video_crowd_count import video_crowd_count
# from datasets import * 

data_mode = cfg.DATASET
datasetting = import_module(f'datasets.setting.{data_mode}')
cfg_data = datasetting.cfg_data

m = video_crowd_count(cfg,cfg_data)

img1 = torch.rand((4,3,768//2,1024//2)).cuda()
m(img1,None)
# for ep in range(epoch):
# 	for i, frame1, frame2,gt_mask1, gt_mask2,den1,den2 in enumearate(carla_simulator):

