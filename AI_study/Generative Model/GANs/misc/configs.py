import os

TRAIN_BATCH_SIZE = 1
INFER_BATCH_SIZE = 8


IMG_WIDTH    = 256
IMG_HEIGHT   = 256
IMG_CHANNELS = 3


LR              = 2e-4
EPOCHS          = 50
STEPS_PER_EPOCH = 800


BASE_OUTPUT_PATH = 'outputs'
GENERATOR_MODEL  = os.path.join(BASE_OUTPUT_BASE, 'models', 'generator')

BASE_IMAGES_PATH = os.path.join(BASE_OUTPUT_PATH, 'images')
GRID_IMAGE_PATH  = os.path.join(BASE_IMAGES_PATH, 'grid.png')