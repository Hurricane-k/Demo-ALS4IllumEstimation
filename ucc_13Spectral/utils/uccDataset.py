try:
    from utils.img_utils import *
    from utils.augmentation import *
    from config.model_params import *
except:
    from ucc_13Spectral.utils.img_utils import *
    from ucc_13Spectral.utils.augmentation import *
    from ucc_13Spectral.config.model_params import *
import json
import os
from torch.utils.data import Dataset
from glob import glob
from sklearn.model_selection import KFold
import numpy as np
import kornia

EPS = 1e-09

class uccDataset(Dataset):

    def __init__(
            self,
            root_dir,
            extension = ['npz','npy'],
            mode = 'train',
            fold = 0,
            transform = None,
            ):
        """_summary_

        Args:
            root_dir (_str_): _root dataset path_
            extension (list, optional): _suffix of dataset_. Defaults to ['npz','npy'].
            mode (str, optional): 'train' or 'test'. Defaults to 'train'.
            fold (int, optional): [0, 1, 2]. Defaults to 0.
            transform (_type_, optional): for data augment (todo-list). Defaults to None.
        """

        self.extension = extension
        self.fold = fold # 0, 1, 2
        self.transform = transform
        self.bin_num = CustomParams.bin_num
        self.boundary_value = CustomParams.boundary_value

        u_coord, v_coord = get_uv_coord(
            self.bin_num,
            range=self.boundary_value * 2
        )  # uv could be negative, range from -boundary_value to +boundary_value
        uv = torch.stack([u_coord, v_coord], dim=-1)
        self.coords_map = (uv + self.boundary_value) / (2 * self.boundary_value)
        self.rgb_map = log_uv_to_rgb_torch(uv)

        self.rgb_map /= (torch.norm(self.rgb_map, 
                                    dim=-1, keepdim=True, 
                                    dtype=self.rgb_map.dtype) + EPS)
        
        pathCV = '/'.join([root_dir,'cvfold.json'])
        if os.path.exists(pathCV):
            pass
        else:
            self.generate_cvfold(root_dir)

        with open(pathCV,'r') as json_file:
            dict_cvfold = json.load(json_file)
        del json_file

        if mode == 'train':
            self.lis_name = dict_cvfold[str(self.fold)]['tri']
        else:
            self.lis_name = dict_cvfold[str(self.fold)]['val']

        self.lis_inp = ['/'.join([root_dir,
                                  'numpy_data',
                                  name+'.'+self.extension[0]]) 
                                  for name in self.lis_name]

        self.lis_out = ['/'.join([root_dir,
                                  'numpy_labels',
                                  name+'.'+extension[1]]) 
                                  for name in self.lis_name]
        
    def __len__(self):
        return len(self.lis_name)
    
    def __getitem__(self, index):
        img = np.load(self.lis_inp[index])['pureStatsMap']
        spc = np.load(self.lis_inp[index])['spectral13']
        ill = np.load(self.lis_out[index])
        nam = self.lis_name[index]

        img = torch.from_numpy(img)
        spc = torch.from_numpy(spc)
        ill = torch.from_numpy(ill)

        # normalize the multi-spectral information
        spc /= (spc.sum() + 1e-9)
        spc = torch.sqrt(spc)
        d_c = spc.shape[0] # the dimension of channel
        spc = spc.view(d_c,1,1)

        img = img.permute(2, 0, 1)
        ret = {'illum': ill, 
               'input': img,
               'spect': spc,
               'name': nam}
        if self.transform is not None:
            ret = self.transform(ret)
        ret = self.extract_feature(ret)
        return ret
    
    def extract_feature(self,ret):
        img = ret['input']
        illum = ret['illum']
        normlised_illum = illum / (torch.norm(illum, 
                                              dim=-1, 
                                              keepdim=True, 
                                              dtype=illum.dtype) + EPS)

        hist = compute_uv_histogram_torch(
            img,
            self.bin_num,
            self.boundary_value,
            channel_first=True,
        )

        hist = torch.unsqueeze(hist, dim=0)
        ret['input'] = hist

        if CustomParams.edge_info:
            edge_img = kornia.filters.sobel(img.unsqueeze(0)).squeeze()
            edge_hist = compute_uv_histogram_torch(
                edge_img,
                self.bin_num,
                self.boundary_value,
                channel_first=True)

            edge_hist = torch.unsqueeze(edge_hist, dim=0)
            ret['input'] = torch.cat([ret['input'], edge_hist], dim=0)

        #%%
        error_map = self.rgb_map @ normlised_illum
        error_map = torch.clamp(error_map, -0.999999, 0.999999)
        error_map = torch.arccos(error_map)

        if CustomParams.coords_map:
            ret['input'] = torch.cat(
                [ret['input'], 
                 self.coords_map.permute(2, 0, 1)], 
                 dim=0)

        ret['target'] = error_map
        return ret
    
        
    @staticmethod
    def generate_cvfold(pathDataset: str, 
                        n_splits:int =3,
                        ex_inp: str = 'npz',
                        ex_out: str = 'npy'):
        """ generate cv_folds

        Args:
            pathDataset (_str_): _the root path of dataset_
            n_splits (int, optional): _folds_. Defaults to 3.
        """
        # import data
        pathinp = '/'.join([pathDataset,'numpy_data'])
        pathout = '/'.join([pathDataset,'numpy_labels'])
        lis_inp = glob('/'.join([pathinp,'*.'+ex_inp]))
        lis_out = glob('/'.join([pathout,'*.'+ex_out]))
        assert len(lis_inp) == len(lis_out), 'inputs mismatch outputs'
        lis_name = [os.path.basename(path_inp).split('.')[0] for path_inp in lis_inp]

        num = len(lis_inp)
        num = np.arange(num)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        train_test = {}
        for idx, data in enumerate(kf.split(num)):
            train, test = data
            train_test[idx] = {}
            train_test[idx]['tri'] = np.array(lis_name)[train].tolist()
            train_test[idx]['val'] = np.array(lis_name)[test].tolist()

        json_name = '/'.join([pathDataset,'cvfold.json'])
        with open(json_name, 'w') as json_file:
            json.dump(train_test, json_file, indent=4)

