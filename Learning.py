#!/usr/bin/env python
# coding: utf-8

# ## 1. Setup configuration

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pds
import numpy as np
import os

from argparse import ArgumentParser
import gc ; gc.enable()

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy,MeanAbsolutePercentageError
from pytorch_forecasting import SMAPE

import joblib

from src.config import load_default

config = load_default()

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    config.gpu_numb = [n_gpu-1]


device = torch.device('cuda:{}'.format(config.gpu_numb[0]))


# ## 2. Load dataset

# In[2]:


raw_df = pds.read_csv(config.dataset_path)

if 'sample_data' in config.dataset_path:
    raw_df = raw_df.drop(columns = 'date')
    


# ##### If you prepared the custom validation datasets, you should skip the below cell.

# In[3]:


nsamples = len(raw_df)
tr_vl_pivot = int(nsamples * 5 / 6)

train_df = raw_df.iloc[:tr_vl_pivot]
valid_df = raw_df.iloc[tr_vl_pivot:]


# **If you ignore the standard scaling, please skip the below cell.</br>**
# **Note that employing the scaler improved the performance of NATMs.**

# In[4]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_df = scaler.fit_transform(train_df)
valid_df = scaler.transform(valid_df)


# ##### Seqeunces slicing

# In[5]:


train_seqs = np.lib.stride_tricks.sliding_window_view(
    x = train_df,
    window_shape = (config.input_length + config.output_length),
    axis = 0
).transpose([0,2,1])

valid_seqs = np.lib.stride_tricks.sliding_window_view(
    x = valid_df,
    window_shape = (config.input_length + config.output_length),
    axis = 0
).transpose([0,2,1])

train_seqs.shape, valid_seqs.shape


# ## 3. Prepare the NATMs 

# In[6]:


from src.models.natm import pl_natm
from src.data_prepare import pl_DataModule

dataset_dict = dict(
    train = (train_seqs[:,:-1], train_seqs[:,-1]),
    valid = (valid_seqs[:,:-1], valid_seqs[:,-1])
)

pldm = pl_DataModule(dataset_dict, config)


# In[7]:


config.input_feature = raw_df.shape[1]
config.output_feature = raw_df.shape[1]


model = pl_natm(config)


# In[8]:


model.train()


save_name = '_'.join([config.exp_name, config.natm_type, config.dataset_path.split('/')[-1][:-4]])

callbacks = [
    ModelCheckpoint(
        dirpath = os.path.join('.',config.save_ckpt_dirs, save_name),
        filename = '{epoch:03d}-{val_loss:.3f}-{val_SMAPE:.3f}',
        save_last = True,
        save_top_k = config.save_top_k,
        monitor = 'val_loss',
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=config.ealry_stop_round,
    )
]

trainer = pl.Trainer(
    enable_progress_bar = config.prog_bar,
    max_epochs = config.epochs, 
    callbacks = callbacks,
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
)


# ## 4. Training

# In[9]:


trainer.fit(model,datamodule = pldm)



# ## 5. Logging

# In[10]:


import joblib


model.eval()
outputs = trainer.predict(model, pldm.val_dataloader())

return_trues = []
return_preds = []

for yt, yp, met, w in outputs:
    return_trues.append(yt.numpy())
    return_preds.append(yp.numpy())

return_trues = scaler.inverse_transform(np.concatenate(return_trues))
return_preds = scaler.inverse_transform(np.concatenate(return_preds))

joblib.dump(
    dict(
        scaler = scaler,
        config = config,
        columns = raw_df.columns
    ), os.path.join('.',config.save_ckpt_dirs, save_name, 'log.joblib')
)


joblib.dump(
    dict(
        train = train_seqs,
        valid = valid_seqs,
    ), os.path.join('.',config.save_ckpt_dirs, save_name, 'data_samples.joblib')
)


# ## 6. Evaluation

# In[11]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def fn_smape(y,y_pred):
    y, y_pred = y, y_pred
    return ((2 * np.abs(y - y_pred)) / (np.abs(y) + np.abs(y_pred))).mean()


def compute_metric(y_true,y_pred):
    matric = r2_score(y_true,y_pred), mean_squared_error(y_true,y_pred)**.5, mean_absolute_error(y_true,y_pred), fn_smape(y_true,y_pred)
    return 'R2:{:.5f} RMSE:{:.5f} MAE:{:.5f} SMAPE:{:.5f}'.format(*matric)



compute_metric(return_trues, return_preds)


# In[12]:


os.path.join('.',config.save_ckpt_dirs, save_name, 'data_samples.joblib')

