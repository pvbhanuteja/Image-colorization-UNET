from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from torchvision import datasets
from skimage.color import rgb2lab, lab2rgb

from torchvision.transforms.functional import resize

class ColorizeData(datasets.ImageFolder):
    def __init__(self,lab_version,transform,**kw):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        # self.input_transform = T.Compose([T.ToTensor(),
        #                                   T.Resize(size=(256,256)),
        #                                   T.Grayscale(),
        #                                   T.Normalize((0.5), (0.5))
        #                                   ])
        # # Use this on target images(colorful ones)
        # self.target_transform = T.Compose([T.ToTensor(),
        #                                    T.Resize(size=(256,256)),
        #                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        self.lab_version=lab_version
        print(transform,lab_version)
        self.transformation = transform
        self.target_transform = None
        super(ColorizeData, self).__init__(**kw)


    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        # if self.transform is not None:
        # self.transform = T.Compose([
        #         T.Resize(size=(256,256)),
        #         ])
        img_original = self.transformation(img)
        img_original = np.asarray(img_original)
        org_img = img_original
        
        if self.lab_version == 1:
            # output is in range [1,1] -> tanh activation
            img_lab = rgb2lab(img_original / 255.0)
            img_lab = (img_lab + [0, 0, 0]) / [100, 128, 128]
        elif self.lab_version == 2:
            # output is in range [0,1]
            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + [0, 128, 128]) / [100, 255, 255]
        else:
            raise ValueError('Incorrect Lab version!!!')
            
        img_ab = img_lab[:,:,1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
        
        img_gray = img_lab[:,:,0]
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(type(org_img),type(img_ab))
        return img_gray, img_ab, org_img
        