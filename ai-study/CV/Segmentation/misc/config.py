import os

import torch

SEP       = os.path.sep
ROOT_PATH = SEP.join(os.getcwd().split(SEP)[:-4])
DATA_PATH = f'{ROOT_PATH}/Datasets/segmentation'

TRAIN_IMAGE_PATH = f'{DATA_PATH}/train/images'
TRAIN_MASK_PATH  = f'{DATA_PATH}/train/masks'

TEST_IMAGE_PATH  = f'{DATA_PATH}/test/images'

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = True if DEVICE == torch.device('cuda') else False

N_CHANNELS = 1
N_CLASSES  = 1
N_LEVELS   = 3

INIT_LR    = 1e-3
EPOCHS     = 100
BATCH_SIZE = 64

INPUT_IMAGE_WIDTH  = 128
INPUT_IMAGE_HEIGHT = 128

THRESHOLD   = 0.5
BASE_OUTPUT = 'result'

MODEL_PATH  = f'{BASE_OUTPUT}/unet_tgs_salt.pth'
PLOT_PATH   = f'{BASE_OUTPUT}/plot.png'
TEST_PATHS  = f'{BASE_OUTPUT}/test_paths.txt'

os.makedirs(BASE_OUTPUT, exist_ok = True)