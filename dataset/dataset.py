from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import json
import torch
import random
from imgaug import augmenters as iaa
from utils import RandomCrop, FDA_source_to_target
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode

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
            img = T.Resize(self.image_size, Image.BILINEAR)(img)
            lbl = T.Resize(self.image_size, Image.NEAREST)(lbl)
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


class FDADataset(Dataset):
    def __init__(self, source, target, path, image_size, augment_data, task, beta, ranking=None):
        """
        Inputs:
        - source: source dataset, "Cityscapes" or "GTA5"
        - target: target dataset, "Cityscapes" or "GTA5"
        - path_source: the base folder of the datasets
        - image_size: a tuple indicating the resizing for the image in the format (Height, Width)
        - augment_data: if to augment data or not
        - task: a string in ['train', 'val']
        - beta: parameter of FDA transformation
        - ranking: if to apply a ranking in the choice of the target image in FDA
        """
        self.source = source
        self.target = target
        self.path_source = os.path.join(path, source)
        self.path_target = os.path.join(path, target)
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
        self.ranking = ranking
        self.beta = beta
        if ranking == "labels":
            if os.path.exists(f"./labels_{self.image_size[0]}_{self.image_size[1]}.pt"):
                lbl_target = torch.load(f"./labels_{self.target}_{self.image_size[0]}_{self.image_size[1]}.pt")
            else:
                lbl_target = self.read_labels()
            self.target_source_index = [self.most_similar_label(lbl_source, lbl_target) for lbl_source in self.lbl_source_names]
        else:
            self.target_source_index = np.arange(0, len(self.img_source_names))


    def load_labels_map(self):
        with open(os.path.join(self.path_source, 'info.json'), 'r') as f:
            self.labels_map = np.array(json.load(f)['label2train'], dtype=np.uint8)[:, 1]

    def load_cityscapes(self, path, task):
        format_path = lambda path: path.split('/')[1][:-1]
        with open(os.path.join(path, f'{task}.txt')) as f:
            img_entry_names = [format_path(i) for i in f.readlines()]
            lbl_entry_names = [i.replace('leftImg8bit', 'gtFine_labelIds') for i in img_entry_names]
        return img_entry_names, lbl_entry_names

    def load_gta5(self, path, task):
        with open(os.path.join(path, f'{task}.txt')) as f:
            img_entry_names = [n[:-1] for n in f.readlines()]
        return img_entry_names, img_entry_names

    def load_paths(self):
        if self.source == "Cityscapes":
            self.img_source_names, self.lbl_source_names = self.load_cityscapes(self.path_source, self.task)
            self.img_target_names, self.lbl_target_names = self.load_gta5(self.path_target, "train")
        else:
            self.img_source_names, self.lbl_source_names = self.load_gta5(self.path_source, self.task)
            self.img_target_names, self.lbl_target_names = self.load_cityscapes(self.path_target, "train")
    
    def read_labels(self):
      # format_path = lambda path: path.split('/')[1][:-1]
      # with open(os.path.join(dataset_folder, 'train.txt')) as f: # os.path.join(self.path, f'{self.task}.txt')) as f:
      #     labels_list = [i.replace('leftImg8bit', 'gtFine_labelIds') for i in [format_path(j) for j in f.readlines()]]
      resize = T.Resize(self.image_size, InterpolationMode.NEAREST)
      labels_concat = None
      for i, label_path in enumerate(self.lbl_target_names):
          label = read_image(os.path.join(self.path_target, "labels", label_path))
          label_resized = resize(label)
          if labels_concat is None:
              labels_concat = label_resized
          else:
              labels_concat = torch.cat((labels_concat, label_resized), 0)
      torch.save(labels_concat, f"./labels_{self.target}_{self.image_size[0]}_{self.image_size[1]}.pt")
      return labels_concat

    def most_similar_label(self, lbl_file_name, lbl_target):
      src_label = read_image(os.path.join(self.path_source, 'labels', lbl_file_name))
      resize = T.Resize(self.image_size, InterpolationMode.NEAREST)
      src_label_resized = resize(src_label)
      index_max = torch.argmax(torch.sum(lbl_target == src_label_resized, (1,2)))
      return index_max

    def __getitem__(self, index):
        seed = random.random()
        img_file_name = self.img_source_names[index]
        lbl_file_name = self.lbl_source_names[index]
        img_target_file_name = self.img_target_names[self.target_source_index[index]]
        img_path = os.path.join(self.path_source, 'images', img_file_name)
        lbl_path = os.path.join(self.path_source, 'labels', lbl_file_name)
        img_target_path = os.path.join(self.path_target, 'images', img_target_file_name)

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        img_target = Image.open(img_target_path)
        if self.task == 'train' and self.data_augmentation:
            # randomly scale image, label and random crop
            scale = random.choice(self.scale)
            scale = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))
            img = T.Resize(scale, Image.BILINEAR)(img)
            lbl = T.Resize(scale, Image.NEAREST)(lbl)
            img_target = T.Resize(scale, Image.BILINEAR)(img_target)
            img = RandomCrop(self.image_size, seed, pad_if_needed=True)(img)  
            lbl = RandomCrop(self.image_size, seed, pad_if_needed=True)(lbl)
            img_target = RandomCrop(self.image_size, seed, pad_if_needed=True)(img_target)  
        else:
            img = T.Resize(self.image_size, Image.BILINEAR)(img)
            lbl = T.Resize(self.image_size, Image.NEAREST)(lbl)
            img_target = T.Resize(self.image_size, Image.BILINEAR)(img_target)
        img = np.array(img)
        lbl = np.array(lbl)
        img_target = np.array(img_target)
        # augment image and label
        if self.task == 'train':# and self.data_augmentation:
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            lbl = seq_det.augment_image(lbl)
            img_target = seq_det.augment_image(img_target)

        # image -> [C, H, W]
        img = Image.fromarray(img)
        img = self.to_tensor(img).float()

        img_target = Image.fromarray(img_target)
        img_target = self.to_tensor(img_target).float()

        img = FDA_source_to_target(torch.unsqueeze(img, dim=0), torch.unsqueeze(img_target, dim=0), L = self.beta)[0]

        lbl_to_numpy = self.labels_map[np.array(lbl, dtype=np.uint8)]
        lbl = torch.from_numpy(lbl_to_numpy).long()

        return img, lbl

    def __len__(self):
        return len(self.img_source_names)
