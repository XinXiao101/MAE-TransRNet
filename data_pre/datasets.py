import os
import torch
from torch.utils.data import Dataset
import numpy as np


class CardiacDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.path = data_path
        self.files = os.listdir(data_path)
        self.transforms = transforms

    def __getitem__(self, index):
        file = self.files[index]

        data = np.load(os.path.join(self.path, file))
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)

        x, y = x[None, ...], y[None, ...]
        # print(x.shape)
        # print(y.shape)
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x) # [B,C,H,W,D]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        return x, y

    def __len__(self):
        return len(self.files)


class CardiacInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.path = data_path
        self.files = os.listdir(data_path)
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        file = self.files[index]

        data = np.load(os.path.join(self.path, file))
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)
        x_Mask = data['x_Mask'].astype(np.float32)
        y_Mask = data['y_Mask'].astype(np.float32)

        x, y = x[None, ...], y[None, ...]
        x_Mask, y_Mask= x_Mask[None, ...], y_Mask[None, ...]

        x, x_Mask = self.transforms([x, x_Mask])
        y, y_Mask = self.transforms([y, y_Mask])

        x = np.ascontiguousarray(x) # [B,C,H,W,D]
        y = np.ascontiguousarray(y)
        x_Mask = np.ascontiguousarray(x_Mask)  # [Bsize,channelsHeight,,Width,Depth]
        y_Mask = np.ascontiguousarray(y_Mask)

        x, y, x_Mask, y_Mask = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_Mask), torch.from_numpy(y_Mask)

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        x_Mask = (x_Mask - x_Mask.min()) / (x_Mask.max() - x_Mask.min())
        y_Mask = (y_Mask - y_Mask.min()) / (y_Mask.max() - y_Mask.min())
        return x, y, x_Mask, y_Mask

    def __len__(self):
        return len(self.files)