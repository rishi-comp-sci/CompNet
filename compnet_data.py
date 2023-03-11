import numpy as np 
import random
import json
import os
import shutil
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Normalize, Resize, Compose, ToTensor, RandomCrop
import sys
from torch.utils.data import Dataset, DataLoader
from random import randint
import random

class compnet_dataset(Dataset):
    def __init__(self, phase='train', root=None, enc_root=None, size=(256, 256), transform=True, v=True):
        if root == None:
            if phase == 'train': root = "../../imagenet/ILSVRC/Data/CLS-LOC/train/"
            elif phase == 'val': root = "../../imagenet/ILSVRC/Data/CLS-LOC/val/"
            elif phase == 'test': root = "../../imagenet/ILSVRC/Data/CLS-LOC/test/"
        self.root = root
        if enc_root == None:
            if phase == 'train': enc_root = "../../imagenet/CLIP_ENCODINGS/train/"
            elif phase == 'val': enc_root = "../../imagenet/CLIP_ENCODINGS/val/"
            elif phase == 'test': enc_root = "../../imagenet/CLIP_ENCODINGS/test/"
        self.enc_root = enc_root
        if(phase!='train'):
            self.files = glob(self.root+"*.JPEG")
        else:
            folders = os.listdir(root)
            self.files = []
            for folder in folders:
                fnames = glob(self.root+folder+"/"+"*.JPEG")
                self.files += fnames
        random.shuffle(self.files)
        self.files = self.files[:min(len(self.files), 130000)]
        self.phase = phase
        self.size = size
        self.aug = self.get_transforms(self.phase, self.size)
        self.transform = transform

        if(v):
            print("Num files for " + phase + " = " + str(len(self.files)))

    def get_fname(self, path):
        return path.split('/')[-1].split('.')[0]
    
    def get_enc_path(self, path):
        return self.enc_root+self.get_fname(path)+".pt"

    def __getitem__(self, idx):
        # print(idx)
        # if t == None: t = torch.tensor([float(randint(0, 1000))]).cpu().type(torch.float32)
        # image = self.aug(Image.open(self.files[idx])).cpu()
        # clip_enc = torch.load(self.get_enc_path(self.files[idx]), map_location='cpu')[0]
        # return image, t, clip_enc

        t = randint(0, 999)
        image = self.aug(Image.open(self.files[idx]).convert("RGB").resize(self.size, resample=Image.ANTIALIAS)).cpu()
        clip_enc = torch.load(self.get_enc_path(self.files[idx]), map_location='cpu')[0]
        return image, t, clip_enc
    
    def get_transforms(self, phase, size):
        data_transforms = [
            # transforms.Resize(size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Scales data into [0,1]
            transforms.Lambda(lambda t: (t*2) - 1) # Scale between [-1, 1]
        ]
        AUG = transforms.Compose(data_transforms)
        
        return AUG
    
    def __len__(self):
        return len(self.files)   
