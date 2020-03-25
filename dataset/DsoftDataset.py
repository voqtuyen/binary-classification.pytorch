import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import pandas as pd


class DsoftDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        img_name = os.path.join(self.root_dir, img_name)
        img_lbl = self.data.iloc[index, 1].astype(np.float32)

        img = Image.open(img_name)

        if self.transform:
            img = self.transform(img)

        return img, img_lbl

    def __len__(self):
        return len(self.data)
