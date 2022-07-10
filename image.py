import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2


def load_data(img_path, train):
    gt_path = img_path.replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    target = Image.open(gt_path).convert('1')

    if train:
        #crop_size = (img.size[0] // 2, img.size[1] // 2)
        crop_size = (256,256)
        if random.randint(0, 9) <= 4.5:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))

    else:
        crop_size = ((img.size[0] //16)*16, (img.size[1] // 16)*16)

        if random.randint(0, 9) <= 4.5:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))

        
    return img, target
    
    

    
    
    
    
    
    
    
    
    