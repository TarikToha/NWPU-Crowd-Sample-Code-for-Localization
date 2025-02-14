import time

from easydict import EasyDict as edict

# init
__C = edict()
cfg = __C

# ------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reproduction
__C.DATASET = 'NWPU'  # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD

# __C.NET = 'RAZ_loc'  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet
# __C.NET = 'LC_Net'  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet
# __C.NET = 'LC_Net_v2'  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet
__C.NET = 'LC_Net_v3'  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet

__C.PRE_GCC = False  # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = 'path to model'  # path to model

__C.RESUME = True  # continue training
# __C.RESUME_PATH = './exp/04-25_05-19_NWPU_RAZ_loc_1e-05/latest_state.pth'  #
# __C.RESUME_PATH = './exp/05-04_23-54_NWPU_LC_Net_1e-05/latest_state.pth'  #
# __C.RESUME_PATH = './exp/07-10_10-54_NWPU_LC_Net_v2_1e-05/latest_state.pth'  #
__C.RESUME_PATH = './exp/07-22_15-07_NWPU_LC_Net_v3_1e-05/latest_state.pth'  #

__C.GPU_ID = [0]  # sigle gpu: [0], [1] ...; multi gpus: [0,1]

# learning rate settings
__C.LR = 1e-5  # learning rate
__C.LR_DECAY = 1  # decay rate
__C.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 2000

# print
__C.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
               + '_' + __C.DATASET \
               + '_' + __C.NET \
               + '_' + str(__C.LR)

__C.EXP_PATH = './exp'  # the path of logs, checkpoints, and current codes

# ------------------------------VAL------------------------
__C.VAL_DENSE_START = 30
__C.VAL_FREQ = 5  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

# ------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes

# ================================================================================
# ================================================================================
# ================================================================================
