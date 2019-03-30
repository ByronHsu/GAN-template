import os
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL
from loader.base_data_loader import BaseDataLoader

def ensure_img(file):
    return file.endswith('.jpg') or file.endswith('.png')

class Dataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.img_root = os.path.join(data_dir)
        self.img_path = [x for x in os.listdir(self.img_root) if ensure_img(x)]
        self.transform = transform
    def __getitem__(self, index):
        img = PIL.Image.open(os.path.join(self.img_root, self.img_path[index])).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.img_path)

class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, opt, training=True):
        transform = transforms.Compose([
            transforms.Resize((opt.img_height, opt.img_width)), 
            transforms.ToTensor()
        ])
        self.dataset = Dataset(data_dir, transform)
        super(DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


