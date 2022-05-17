from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import json
import torch
import random
from imgaug import augmenters as iaa
from utils import RandomCrop

class SegmentationDataset(Dataset):
    def __init__(self, path, image_size, augment_data, task):
        """
        Inputs:
        - path: the base folder of the dataset
        - image_size: a tuple indicating the resizing for the image in the format (Height, Width)
        - task: a string in ['train', 'val']
        """
        self.path = path
        self.image_size = image_size
        self.task = task
        self.scale = [0.75, 1, 1.5, 1.75, 2] # scales from BiSeNet paper
        self.fliplr = iaa.Fliplr(0.5)
        # mean and std from ImageNet
        img_net_normalization = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.to_tensor = T.Compose([T.ToTensor(), img_net_normalization] if augment_data else [T.ToTensor()])
        self.data_augmentation = augment_data
        self.load_labels_map()
        self.load_paths()

    def load_labels_map(self):
        with open(os.path.join(self.path, 'info.json'), 'r') as f:
            self.labels_map = np.array(json.load(f)['label2train'], dtype=np.uint8)[:, 1]

    def load_paths(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        seed = random.random()
        img_file_name = self.img_entry_names[index]
        lbl_file_name = self.lbl_entry_names[index]
        img_path = os.path.join(self.path, 'images', img_file_name)
        lbl_path = os.path.join(self.path, 'labels', lbl_file_name)

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        if self.task == 'train' and self.data_augmentation:
            # randomly scale image, label and random crop
            scale = random.choice(self.scale)
            scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))
            img = T.Resize(scale, Image.BILINEAR)(img)
            lbl = T.Resize(scale, Image.NEAREST)(lbl)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)  
            lbl = RandomCrop(self.image_size, seed, pad_if_needed=True)(lbl)  
        else:
            img = T.Resize(self.image_size)(img)
            lbl = T.Resize(self.image_size)(lbl)
        img = np.array(img)
        lbl = np.array(lbl)
        # augment image and label
        if self.task == 'train':# and self.data_augmentation:
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            lbl = seq_det.augment_image(lbl)

        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        lbl_to_numpy = self.labels_map[np.array(lbl, dtype=np.uint8)]
        lbl = torch.from_numpy(lbl_to_numpy).long()

        return img, lbl

    def __len__(self):
        return len(self.img_entry_names)

class Cityscapes(SegmentationDataset):
    def load_paths(self):
        format_path = lambda path: path.split('/')[1][:-1]
        with open(os.path.join(self.path, f'{self.task}.txt')) as f:
            self.img_entry_names = [format_path(i) for i in f.readlines()]
            self.lbl_entry_names = [i.replace('leftImg8bit', 'gtFine_labelIds') for i in self.img_entry_names]

class Gta5 (SegmentationDataset):
    def load_paths(self):
        with open(os.path.join(self.path, f'{self.task}.txt')) as f:
            self.img_entry_names = [n[:-1] for n in f.readlines()]
            self.lbl_entry_names = self.img_entry_names 
