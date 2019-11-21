import random
from pathlib import Path

import numpy as np
import skimage
import torch
from skimage import io
from torch.nn import UpsamplingBilinear2d
from torch.utils.data import Dataset


def lr_sampler(lr, lr_mask, num_k, top=False):
    arr_percent_mask = lr_mask.sum(1).sum(1) / (128 * 128)  # Percentage of masked pixels
    lr = lr[arr_percent_mask > 0]  # Selecting only the frame that are not empty
    assert num_k < lr.shape[0]  # num_k must be < 9
    if top:
        argsort_arr = np.argsort(arr_percent_mask)[::-1]
        lr = lr[argsort_arr[:num_k]]

    idx = np.random.choice(lr.shape[0], num_k)
    return lr[idx]


def lr_augment(lr, lr_mask): # Compute the mean and the stat of LR images
    arr_percent_mask = lr_mask.sum(1).sum(1) / (128 * 128)
    aval = lr[arr_percent_mask > 0]
    return np.stack([np.mean(aval, axis=0), np.median(aval, axis=0)])


class ProbaDataset(Dataset):
    def __init__(self, root, train_test_val='train', top_k=5, rand=False, stat=True):
        super().__init__()
        self.dir = Path(root) / 'RED'
        self.list_dir = list(self.dir.iterdir())
        self.dir = Path(root) / 'NIR'
        self.list_dir += list(self.dir.iterdir())

        self.train_test_val = train_test_val

        prop = 0.1
        seed = 123
        random.seed(seed)
        random.shuffle(self.list_dir)
        if train_test_val == 'train':
            self.list_dir = self.list_dir[:-int(prop * len(self.list_dir))]
        elif train_test_val == 'val':
            self.list_dir = self.list_dir[-int(prop * len(self.list_dir)):]

        self.top_k = top_k
        self.rand = rand
        self.stat = stat
        self.upsample = UpsamplingBilinear2d(scale_factor=3) # Baseline

    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, idx):
        dir = self.list_dir[idx]
        X = dir / 'arr_SR.npy'
        mask = dir / 'arr_QM.npy'
        mask = np.load(mask).astype(np.bool)
        X = np.load(X).astype(np.uint16)

        if self.top_k > 0:
            X_ok = lr_sampler(X / np.iinfo(np.uint16).max, mask, num_k=self.top_k, top=self.rand)
        else:
            X_ok = None

        if self.stat:
            X_stat = lr_augment(X / np.iinfo(np.uint16).max, mask)
        else:
            X_stat = None

        if X_stat is None:
            X = X_ok
        elif X_ok is None:
            X = X_stat
        else:
            X = np.concatenate([X_stat, X_ok])

        lr = torch.from_numpy(skimage.img_as_float(X).astype(np.float32))

        name = dir.name
        sr = self.upsample(lr[1:2].unsqueeze(0))[0]

        if self.train_test_val == 'test':
            return lr, name, sr

        HR_im = np.array(io.imread(dir / 'HR.png'), dtype=np.uint16)
        HR_mask = np.array(io.imread(dir / 'SM.png'), dtype=np.bool)

        hr = torch.from_numpy(skimage.img_as_float(HR_im).astype(np.float32))
        mask_hr = torch.from_numpy(HR_mask.astype(np.float32))

        return lr, hr, mask_hr, name, sr
