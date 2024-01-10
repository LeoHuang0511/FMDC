import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed, for reproduction
__C.DATASET = 'HT21'       # dataset selection:  HT21, SENSE
__C.NET = 'VGG16_FPN'     # 'VGG16_FPN'
__C.task = "DEN"  # LAB for colorization , den for training 
#__C.continuous= False # load data continous

__C.RESUME =False# continue training
__C.RESUME_PATH = './exp/CARLA/05-10_01-08_CARLA_VGG16_FPN_5e-05/ep_15_iter_38000_mae_39.045_mse_75.575_seq_MAE_75.984_WRAE_26.019_MIAE_4.175_MOAE_5.109.pth'
__C.GPU_ID = '0'  # sigle gpu: '0'; multi gpus: '0,1

__C.sinkhorn_iterations = 100
__C.FEATURE_DIM = 256
__C.ROI_RADIUS = 4.
if __C.DATASET == 'SENSE':
    __C.VAL_INTERVALS = 8
else:
    __C.VAL_INTERVALS = 50
# learning rate settings
__C.LR_Base = 5e-5 # learning rate densitymap
__C.LR_Thre = 1e-4 #mask branch

__C.LR_DECAY = 0.95
__C.WEIGHT_DECAY = 1e-5  # decay rate
# when training epoch is more than it, the learning rate will be begin to decay

__C.MAX_EPOCH = 500

# print
__C.PRINT_FREQ = 20

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + __C.NET \
    + '_' + str(__C.LR_Base)

__C.VAL_VIS_PATH = './exp/'+__C.DATASET+'_val'
__C.EXP_PATH = os.path.join('./exp', __C.DATASET)  # the path of logs, checkpoints, and current codes
if not os.path.exists(__C.EXP_PATH ):
    os.makedirs(__C.EXP_PATH )
#------------------------------VAL------------------------

if __C.DATASET == 'HT21':
    __C.VAL_FREQ = 1  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ
    __C.VAL_DENSE_START = 2
else:
    __C.VAL_FREQ = 1
    __C.VAL_DENSE_START = 0
#------------------------------VIS------------------------

# ================================================================================
