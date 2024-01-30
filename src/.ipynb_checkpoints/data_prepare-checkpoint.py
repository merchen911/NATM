import pandas as pds
import numpy as np
import os
import re
from typing import Union, List, Tuple, Dict

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


naming = lambda path: re.search(r'/\w+\.',path).group()[1:-1]



class Loader:
    def __init__(
        self,
        config,
        df: pds.DataFrame,
#         n_fold: int = 5
    ) -> None:
        
        self._config = config
        
        self.input_length = config.input_length
        self.output_length = config.output_length
        self.total_length = self.input_length + self.output_length
        
        self.df = df.copy().reset_index(drop = True)
        if 'ETT' in config.dataset_name:
            self.df = self.df.drop(columns = ['date'])
            
        self.columns = self.df.columns
        
        self.n_fold = config.n_fold + 2
        self.scalerx = StandardScaler()
        self.scalery = StandardScaler()

        self.raw_data = dict()
#         self.fold_idx_dict = self.__split_idx__()
        
        
    def __split_idx__(self, n_fold):
        
        dt = self.n_fold + int(10 * self._config.training_size)
        
        split_points = np.linspace(0, self.df.shape[0], dt).astype(int)
        
        train_st_idx = split_points[n_fold]
        valid_st_idx = split_points[n_fold + int(10 * self._config.training_size) + 1]  
        valid_ed_idx = split_points[n_fold + int(10 * self._config.training_size) + 2]
        
        train_idx = np.arange(train_st_idx, valid_st_idx, 1)
        valid_idx = np.arange(valid_st_idx, valid_ed_idx, 1)
#         print(n_fold, n_fold + int(10 * self._config.training_size) + 2, len(split_points), train_st_idx, valid_st_idx, valid_ed_idx, train_idx.shape, valid_idx.shape)
        return train_idx, valid_idx
        
#         fold_idx_dict = dict()
#         for n,(train_idx, valid_idx) in enumerate(
#             self.TSS.split(self.df)):
#             fold_idx_dict[n] = dict(
#                 train = train_idx,
#                 valid = valid_idx[:int(len(valid_idx) / self._config.n_fold * (n+1))]
#             )
#         return fold_idx_dict
            
        
    def __scaling__(self,n_fold):
#         train_idx, valid_idx = self.fold_idx_dict[n_fold].values()
        train_idx, valid_idx = self.__split_idx__(n_fold)
        train_set, valid_set = self.df.iloc[train_idx], self.df.iloc[valid_idx]
        
        train_x = self.scalerx.fit_transform(train_set)
        valid_x = self.scalerx.transform(valid_set)
        
        train_y = self.scalery.fit_transform(train_set)
        valid_y = self.scalery.transform(valid_set)
        
        return dict(
            train = (
                self.__repeat__(train_x)[:,:self.input_length],
                self.__repeat__(train_y)[:,-self.output_length:]
            ),
            valid = (
                self.__repeat__(valid_x)[:,:self.input_length],
                self.__repeat__(valid_y)[:,-self.output_length:]
            ),
        )

    
    def __repeat__(self,arr) -> np.ndarray:
        stacked_arr = []
        for n in range(0,-self.total_length,-1):
            stacked_arr.append(np.roll(arr,n,0))
        stacked_arr = np.stack(stacked_arr,1)
        stacked_arr = stacked_arr[:-self.total_length+1]
        return stacked_arr
    
    
    def __call__(self, n_fold: int) -> dict:
        return self.__scaling__(n_fold)


    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.datax = data[0].astype(np.float32)
        self.datay = data[1].squeeze().astype(np.float32)

    def __len__(self):
        return len(self.datax)

    def __getitem__(self, idx):
        return self.datax[idx],self.datay[idx]


    
class pl_DataModule(pl.LightningDataModule):
    def __init__(self, 
                 datasets,
                 config
                ):
        super().__init__()
        self._config = config
#         self.save_hyperparameters()
        self.batch_size = self._config.batch_size
        self.num_workers = self._config.num_workers
        self.train_dataset = CustomDataset(datasets['train'])
        self.valid_dataset = CustomDataset(datasets['valid'])
        
    def setup(self, stage='fit'):
        if stage=='fit':
            pass

    def train_dataloader(self):
        if self.num_workers > 1:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                              shuffle=True, num_workers = self.num_workers) # NOTE : Shuffle
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle
        
    def val_dataloader(self):
        if self.num_workers > 1:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
        else:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.num_workers > 1:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
        else:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size)
    
    
    
class Informer_CustomDataset(Dataset):
    def __init__(self, data, config):
        self._config = config
        self.datax = data[0]
        self.datay = data[1].squeeze()

        self._config.seq_len = self._config.input_length
        self._config.label_len = self._config.input_length - config.output_length
        self._config.pred_len = self._config.output_length
        
        
    def __len__(self):
        return len(self.datax)

    def __getitem__(self, idx):
        x = self.datax[idx].astype(np.float32)
        y = self.datay[idx].astype(np.float32)
        
        empty = np.zeros((self._config.pred_len,x.shape[1])).astype(np.float32)
        x_dec = np.concatenate([x[-self._config.label_len:],empty])
        
        return (x,x_dec),y


class pl_Informer_DataModule(pl.LightningDataModule):
    def __init__(self, 
                 datasets,
                 config
                ):
        super().__init__()
        self._config = config
#         self.save_hyperparameters()
        self.batch_size = self._config.batch_size
        self.num_workers = self._config.num_workers
        self.train_dataset = Informer_CustomDataset(datasets['train'], self._config)
        self.valid_dataset = Informer_CustomDataset(datasets['valid'], self._config)

        
    def setup(self, stage='fit'):
        if stage=='fit':
            pass
        
    def train_dataloader(self):
        if self.num_workers > 1:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                              shuffle=True, num_workers = self.num_workers) # NOTE : Shuffle
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle
        
    def val_dataloader(self):
        if self.num_workers > 1:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
        else:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.num_workers > 1:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
        else:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size)
    
    
class toy_Loader:
    def __init__(
        self,
        config,
        arr,
        n_fold: int = 10
    ) -> None:
        
        self._config = config
        self.input_length = config.input_length
        
        self.output_length = config.output_length
        self.total_length = self.input_length + self.output_length
        
        self.arr = arr.copy()
        
        self.n_fold = config.n_fold + 2
        self.scalerx = StandardScaler()
        self.scalery = StandardScaler()

        self.raw_data = dict()
#         self.fold_idx_dict = self.__split_idx__()
        
        
    def __split_idx__(self, n_fold):
        
        dt = self.n_fold + int(10 * self._config.training_size)
        
        split_points = np.linspace(0, self.df.shape[0], dt).astype(int)
        
        train_st_idx = split_points[n_fold]
        valid_st_idx = split_points[n_fold + int(10 * self._config.training_size) + 1]  
        valid_ed_idx = split_points[n_fold + int(10 * self._config.training_size) + 2]
        
        train_idx = np.arange(train_st_idx, valid_st_idx, 1)
        valid_idx = np.arange(valid_st_idx, valid_ed_idx, 1)
#         print(n_fold, n_fold + int(10 * self._config.training_size) + 2, len(split_points), train_st_idx, valid_st_idx, valid_ed_idx, train_idx.shape, valid_idx.shape)
        return train_idx, valid_idx
        
#         fold_idx_dict = dict()
#         for n,(train_idx, valid_idx) in enumerate(
#             self.TSS.split(self.arr,self.arr)):
#             fold_idx_dict[n] = dict(
#                 train = train_idx,
#                 valid = valid_idx[:int(len(valid_idx) / self._config.n_fold * (n+1))]
#             )
#         return fold_idx_dict
            
        
    def __scaling__(self,n_fold):
#         train_idx, valid_idx = self.fold_idx_dict[n_fold].values()
        train_idx, valid_idx = self.__split_idx__(n_fold)

        train_set, valid_set = self.arr[train_idx], self.arr[valid_idx]
        
        train_x = self.scalerx.fit_transform(train_set)
        valid_x = self.scalerx.transform(valid_set)
        
        train_y = self.scalery.fit_transform(train_set)
        valid_y = self.scalery.transform(valid_set)
        
        return dict(
            train = (
                self.__repeat__(train_x)[:,:self.input_length],
                self.__repeat__(train_y)[:,-self.output_length:]
            ),
            valid = (
                self.__repeat__(valid_x)[:,:self.input_length],
                self.__repeat__(valid_y)[:,-self.output_length:]
            ),
        )

    
    def __repeat__(self,arr) -> np.ndarray:
        stacked_arr = []
        for n in range(0,-self.total_length,-1):
            stacked_arr.append(np.roll(arr,n,0))
        stacked_arr = np.stack(stacked_arr,1)
        stacked_arr = stacked_arr[:-self.total_length+1]
        return stacked_arr
    
    
    def __call__(self, n_fold: int) -> dict:
        return self.__scaling__(n_fold)