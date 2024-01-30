import pandas as pds
import numpy as np
import os
import re
from typing import Union, List, Tuple, Dict

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


    
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
        self.batch_size = self._config.batch_size
        self.num_workers = self._config.num_workers
        self.train_dataset = CustomDataset(datasets['train'])
        self.valid_dataset = CustomDataset(datasets['valid'])
        
        
    def setup(self, stage='fit'):
        if stage=='fit':
            pass

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers = self.num_workers) # NOTE : Shuffle
        
        
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers = self.num_workers)

    
    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    
    
    def teardown(self, stage: str):
        pass
    
    
    def prepare_data(self):
        pass
    
    
    
    