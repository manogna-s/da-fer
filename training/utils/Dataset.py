import os
import copy
import random
import numpy as np
from PIL import Image, ImageDraw

import torch.utils.data as data

def L_loader(path):
    return Image.open(path).convert('L')

def RGB_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(data.Dataset):
    def __init__(self, imgs, labels, bboxs, landmarks, split, domain, transform=None, loader=RGB_loader):
        self.imgs = imgs
        self.labels = labels
        self.bboxs = bboxs
        self.landmarks = landmarks
        self.transform = transform
        self.loader = loader
        self.split = split
        self.domain = domain
        
    def __getitem__(self, index):
        img, label, bbox, landmark = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.bboxs[index]), np.array(copy.deepcopy(self.landmarks[index]))
        ori_img_w, ori_img_h = img.size

        # BoundingBox
        left   = bbox[0]
        top    = bbox[1]
        right  = bbox[2]
        bottom = bbox[3]

        enlarge_bbox = True

        if self.split=='train' and self.domain=='source':
            random_crop = True
            random_flip = True
        else:
            random_crop = False
            random_flip = False

        # Enlarge BoundingBox
        padding_w, padding_h = int(0.5 * max( 0, int( 0.20 * (right-left) ) ) ), int( 0.5 * max( 0, int( 0.20 * (bottom-top) ) ) )
    
        if enlarge_bbox:
            left  = max(left - padding_w, 0)
            right = min(right + padding_w, ori_img_w)

            top = max(top - padding_h, 0)
            bottom = min(bottom + padding_h, ori_img_h)

        if random_crop:
            x_offset = random.randint(-padding_w, padding_w)
            y_offset = random.randint(-padding_h, padding_h)

            left  = max(left + x_offset, 0)
            right = min(right - x_offset, ori_img_w)

            top = max(top + y_offset, 0)
            bottom = min(bottom - y_offset, ori_img_h)

        img = img.crop((left,top,right,bottom))
        crop_img_w, crop_img_h = img.size

        landmark[:,0]-=left
        landmark[:,1]-=top

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:,0] = (right - left) - landmark[:,0]

        # Transform Image
        trans_img = self.transform(img)

        inputSizeOfCropNet = 28
        landmark[:, 0] = landmark[:, 0] * inputSizeOfCropNet / crop_img_w
        landmark[:, 1] = landmark[:, 1] * inputSizeOfCropNet / crop_img_h
        landmark = landmark.astype(np.int)

        grid_len = 7 
        half_grid_len = int(grid_len/2)

        for index in range(landmark.shape[0]):
            if landmark[index,0] <= (half_grid_len - 1):
                landmark[index,0] = half_grid_len
            if landmark[index,0] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index,0] = inputSizeOfCropNet - half_grid_len - 1
            if landmark[index,1] <= (half_grid_len - 1):
                landmark[index,1] = half_grid_len
            if landmark[index,1] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index,1] = inputSizeOfCropNet - half_grid_len - 1 
        
        return trans_img, landmark, label

    def __len__(self): 
        return len(self.imgs)
