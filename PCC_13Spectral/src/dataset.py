
from torch.utils.data import Dataset
import glob
import os
try:
    from src.utils import hwc_to_chw, k_fold, EPS, norm_img
except:
    from PCC_13Spectral.src.utils import hwc_to_chw, k_fold, EPS, norm_img
import numpy as np
from pathlib import Path
try:
    from src.data_aug import AwbAug
except:
    from PCC_13Spectral.src.data_aug import AwbAug

FULL_TEST = False

class CcData(Dataset):
    def __init__(self, path, 
                 mode='semantic_spectral', 
                 train=True, augment=False, fold_num=0):
        """_init_

        Args:
            path (_type_): the path of DataSet, set by argparse
            train (bool, optional): test or train model. Defaults to True.
            fold_num (int, optional): the number of fold (cross-validation). Defaults to 0.
            mode (str, optional): ['semantic_spectral','semantic','spectral','coarse2fine_ucc','coarse2fine_ucc_spec_pcc']
                                  determine the dimension of input
            augment (bool, optional): due to the multi-channel spectral info
                                      multi-channel spectral info cannot be augmented along with img
        """

        self.path = path
        self.train = train
        self.mode = mode
        self.augment = augment

        self.illu_full = glob.glob('/'.join([path,'numpy_labels','*.npy']))
        self.img_full = glob.glob('/'.join([path,'numpy_data','*.npz']))

        # sort file
        self.illu_full.sort(key=lambda x: x.split('\\')[-1].split('_')[-1].split('.')[0])
        self.img_full.sort(key=lambda x: x.split('\\')[-1].split('_')[-1].split('.')[0])

        # cross-validation
        train_test = k_fold(n_splits=3, num=len(self.img_full))
        img_idx = train_test['train' if self.train else 'test'][fold_num]

        # collect data based on CV fold
        self.fold_data = [self.img_full[i] for i in img_idx]
        self.fold_illu = [self.illu_full[i] for i in img_idx]
        self.name_data = [os.path.basename(file_data).split('.')[0] for file_data in self.fold_data]

        # instantiate AwbAug
        self.data_aug = AwbAug(self.illu_full)

    def __len__(self):
        """ necessary for the custom dataset

            Return: the total number of data
        """
        return len(self.fold_data)

    def feature_select(self, img_tmp, 
                       thresh_dark=0.02, 
                       thresh_saturation=0.98):
        """
        The four feature selected, i.e., bright, max, mean and dark pixels
        """
        img_tmp = img_tmp.reshape(-1, 3)
        img_tmp = img_tmp[np.all(img_tmp > thresh_dark, axis=1), :]
        img_tmp = img_tmp[np.all(img_tmp < thresh_saturation, axis=1), :]
        # 0. Brightest pixel
        bright_v = img_tmp[np.argmax(img_tmp.sum(axis=1))]
        # 1. Maximum pixel
        max_wp = img_tmp.max(axis=0)
        # 2. Average pixel
        mean_v = img_tmp.mean(axis=0)
        # 3. Darkest pixel
        dark_v = img_tmp[np.argmin(img_tmp.sum(axis=1))]
        # ---Testing the weight of different features---
        # mask_feature = np.array([0, 0, 0])
        # dark_v = mask_feature
        feature_data = np.vstack([bright_v, max_wp, mean_v, dark_v])
        # feature_num = len(feature_data)
        feature_data /= (feature_data.sum(axis=1).reshape(-1, 1) + EPS)
        feature_data = feature_data[:, :2]

        return feature_data

    def __getitem__(self, idx, thresh_dark=0.02, thresh_saturation=0.98):
        """ Gets next data in the dataloader.

        Note: We pre-processed the input data in the format of '.npy' for fast processing. If
        you want to train your own dataset, the corresponding of loadig image should also be changed.

        Return 
            feature_data (np.array) with the shape of (4,2)
        """

        # load data (notice: spectral info and img are zipped in one .npz)
        img_data = np.load(self.fold_data[idx])['pureStatsMap']
        spc_data = np.load(self.fold_data[idx])['spectral13']
        gd_data = np.load(self.fold_illu[idx])

        # generally, augment data are not concerned
        if self.train and self.augment and self.mode == 'semantic':
            img_data_aug, gd_data_aug = self.data_aug.awb_aug(gd_data, img_data)
        else:
            img_data_aug = norm_img(img_data)
            gd_data_aug = gd_data

        # in case, normalization groundtruth again
        gd_data = gd_data/gd_data.sum()
        gd_data_aug = gd_data_aug/gd_data_aug.sum()
        spc_data = spc_data/spc_data.sum()
        
        # in data_aug.py, change the default value of radius in function circle_point
        # but no need to change the default value of radius in function point
        # I need to change there logic
        img_aug_2D = img_data_aug.reshape(-1, 3)
        img_aug_2D = img_aug_2D[np.all(img_aug_2D > thresh_dark, axis=1), :]
        img_aug_2D = img_aug_2D[np.all(img_aug_2D < thresh_saturation, axis=1), :]

        # if train mode, no need to return file_name
        # if test mode, need to return file_name

        if len(img_aug_2D) > 0: # can use augumented data
            feature_data = self.feature_select(img_data_aug)
            del img_data
            if self.mode == 'semantic':
                return feature_data.astype(np.float32), gd_data_aug.astype(np.float32), self.name_data[idx]
            elif self.mode == 'spectral':
                return spc_data.astype(np.float32), gd_data_aug.astype(np.float32), self.name_data[idx]
            else:
                return np.concatenate([feature_data.flatten(),spc_data]).astype(np.float32), gd_data_aug.astype(np.float32), self.name_data[idx]

        else:
            feature_data = self.feature_select(img_data)
            if self.mode == 'semantic':
                return feature_data.astype(np.float32), gd_data.astype(np.float32), self.name_data[idx]
            elif self.mode == 'spectral':
                return spc_data.astype(np.float32), gd_data.astype(np.float32), self.name_data[idx]
            else:
                return np.concatenate([feature_data.flatten(),spc_data]).astype(np.float32), gd_data.astype(np.float32), self.name_data[idx]

