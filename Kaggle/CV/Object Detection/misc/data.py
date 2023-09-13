import os

from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms
from imutils.paths import *
import albumentations as A
import torch
import cv2


class TurtleDataset(Dataset):


    def __init__(self, image_paths, label_paths, size = 640, transform = None):

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform   = transform
        self.size        = size

    
    def adjust_coords(self, yolo, WH):

        x, y, w, h = yolo
        W, H       = WH

        fix_x1, fix_x2 = (2*x - w) * W / 2, (2*x + w) * W / 2
        fix_y1, fix_y2 = (2*y - h) * H / 2, (2*y + h) * H / 2

        return (int(fix_x1), int(fix_x2), int(fix_y1), int(fix_y2))


    def draw_bboxes(self, idx):

        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image            = cv2.resize(image, (self.size, self.size), cv2.INTER_LINEAR)
        fix_h , fix_w    = image.shape[:2]

        coords         = open(self.label_paths[idx], 'r').readlines()
        boxes          = [list(map(float, coord.split())) for coord in coords]

        rec            = image.copy()
        bboxes, labels = [], [] 
        for box in boxes:
            lb,  x,  y,  w, h = box
            
            ## 이미지 리사이징으로 인한 좌표 수정
            x1, x2, y1, y2    = self.adjust_coords((x, y, w, h), (fix_w, fix_h))
            rec               = cv2.rectangle(rec, (x1, y1), (x2, y2), (255, 255, 0), 3)

            x1, x2, y1, y2    = x1 / fix_w, x2 / fix_w, y1 / fix_h, y2 / fix_h

            bboxes.append((x1, y1, x2, y2))
            labels.append(lb)

        bboxes   = torch.as_tensor(bboxes, dtype = torch.float32)
        labels   = torch.as_tensor(labels, dtype = torch.int64)
        area     = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        is_crowd = torch.zeros((bboxes.shape[0]), dtype = torch.int64)

        target = {}
        target['area']     = area
        target['boxes']    = bboxes
        target['labels']   = labels
        target['iscrowd']  = is_crowd
        target['image_id'] = torch.tensor([idx])

        if self.transform:

            sample = self.transform(image  = image / 255.0,
                                    bboxes = target['boxes'],
                                    labels = labels)

            image           = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
        return rec, image, target 
    
    def __len__(self): return len(self.image_paths)

    
    def __getitem__(self, idx):

        _, image, target = self.draw_bboxes(idx)
        return image, target
        


def get_transform(train):

    if train:
        return A.Compose([
                A.HorizontalFlip(0.5),
                ToTensorV2(p = 1.0)
            ], bbox_params = {'format' : 'pascal_voc', 'label_fields' : ['labels']})

    else:

        return A.Compose([
                ToTensorV2(p = 1.0)
            ], bbox_params = {'format' : 'pascal_voc', 'label_fields' : ['labels']})
        