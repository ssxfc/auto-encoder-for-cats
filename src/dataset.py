import os
import sys
import torch
import math

import numpy as np
from PIL import Image


from torch.utils.data import DataLoader, Dataset


class CatsDataset(Dataset):
    def __init__(self, root_dir, category='train', span=[0, 0.7]):
        super().__init__()
        self.category = category
        self.root_dir = root_dir
        # 存储数据集合
        self.imgs = os.listdir(root_dir)
        total_size = len(self.imgs)
        self.start_idx = math.floor(span[0] * total_size)
        self.end_idx = math.floor(span[1] * total_size)
        self.size = self.end_idx - self.start_idx

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        img_tensor = self._pre_process(self.imgs[self.start_idx + index])
        return img_tensor, img_tensor

    def _pre_process(self, image_path, form='RGB', classify='img'):
        img = Image.open(os.path.join(self.root_dir, image_path)).convert(form)
        np_img = np.array(img)
        if classify == 'img':
            np_img = np_img.transpose((2, 0, 1))
            if (np_img > 1).any():
                np_img = (np_img / 255.0).astype(np.float32)
        else:
            np_img = np_img[np.newaxis, :].astype(np.int64)
        tensor_np_img = torch.as_tensor(np_img)
        return tensor_np_img


    @staticmethod
    def get_dataloader():
        batch_size = 2
        root_dir = r"D:\datasets\cats\Data"
        train_dataset = CatsDataset(root_dir, category='train', span=[0.0, 0.7])
        val_dataset = CatsDataset(root_dir, category='val', span=[0.7, 0.9])
        test_dataset = CatsDataset(root_dir, category='test', span=[0.9, 1.0])
        return DataLoader(train_dataset, batch_size=batch_size), \
            DataLoader(val_dataset, batch_size=batch_size), \
            DataLoader(test_dataset, batch_size=batch_size)