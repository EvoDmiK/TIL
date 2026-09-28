import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from misc import config


def prepare_plot(image, mask, pred):

    fig, axes = plt.subplots(1, 3, figsize = (10, 10))
    images    = [image, mask, pred]
    titles    = ['image', 'ground truth', 'predicted']
    
    for ax, image, title in zip(axes, images, titles):

        ax.imshow(image)
        ax.set_title(title)

    fig.tight_layout()
    fig.show()


def evaluation(model, image_path):

    model.eval()
    with torch.no_grad():

        file_name = image_path.split(os.path.sep)[-1]
        image     = cv2.imread(image_path)
        image     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image     = image.astype('float32') / 255.0

        orig      = image.copy()
        gt_path   = f'{config.TRAIN_MASK_PATH}/{file_name}'
        gt_mask   = cv2.imread(gt_path, 0)
        gt_mask   = cv2.resize(gt_mask, (config.INPUT_IMAGE_WIDTH,
                                         config.INPUT_IMAGE_HEIGHT))
        image     = np.transpose(image, (2, 0, 1))
        image     = np.expand_dims(image, 0)
        image     = torch.from_numpy(image).to(config.DEVICE)

        pred_mask = model(image).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()

        pred_mask = (pred_mask > config.THRESHOLD) * 255
        pred_mask = pred_mask.astype(np.uint8)

        prepare_plot(orig, gt_mask, pred_mask)


        