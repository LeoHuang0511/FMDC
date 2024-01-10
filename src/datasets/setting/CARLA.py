from easydict import EasyDict as edict


# init
__C_CARLA = edict()

cfg_data = __C_CARLA

__C_CARLA.TRAIN_SIZE =(768,1024) # (848,1536) #
__C_CARLA.DATA_PATH = '../dataset/CARLA/'
__C_CARLA.TRAIN_LST = 'train.txt'
__C_CARLA.VAL_LST =  'val.txt'
__C_CARLA.TEST_LST =  'test.txt'

# __C_HT21.MEAN_STD = (
#     [117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.]
# )
__C_CARLA.MEAN_STD = (
    [0.3467, 0.5197, 0.4980], [0.2125, 0.0232, 0.0410]
)


__C_CARLA.DEN_FACTOR = 200.

__C_CARLA.RESUME_MODEL = ''#model path
__C_CARLA.TRAIN_BATCH_SIZE = 2 #  img pairs
__C_CARLA.TRAIN_FRAME_INTERVALS=(40,85)
__C_CARLA.VAL_BATCH_SIZE = 1 # must be 1


