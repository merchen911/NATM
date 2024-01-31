#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import product
import os
from glob import glob
import warnings ; warnings.filterwarnings('ignore')
import gc ; gc.enable()
import joblib

import numpy as np

import torch
import pytorch_lightning as pl

from matplotlib import pyplot as plt
import matplotlib as MP
import seaborn as sbn


# #### Please use your checkpoints path

# In[2]:


log = joblib.load(os.path.join('ckpt','test_Feature_ETTh1','log.joblib'))
data = joblib.load(os.path.join('ckpt','test_Feature_ETTh1','data_samples.joblib'))
ckpt = torch.load(glob(os.path.join('ckpt','test_Feature_ETTh1','epoch*'))[0])
config = log['config']
scaler = log['scaler']
device = torch.device(f'cuda:{config.gpu_numb[0]}')


# In[3]:


data.keys()


# In[4]:


from src.models.natm import pl_natm
from src.data_prepare import pl_DataModule
from sklearn.preprocessing import StandardScaler

### please change to the natms type what you learned before
### config.natm_type = 'Feature' or others in ['Independent','Time']

model = pl_natm(config)
model.to(device)
model.eval()
model.load_state_dict(ckpt['state_dict'])


# In[5]:


train_x, train_y = data['train'][:,:-1], data['train'][:,-1]
valid_x, valid_y = data['valid'][:,:-1], data['valid'][:,-1]


# #### You can visualized the other sample to change the number of 'sample_numb'

# In[6]:


sample_numb = 10



x = valid_x[sample_numb:sample_numb+1]
y = valid_y[sample_numb:sample_numb+1]



with torch.no_grad():
    torch_x = torch.tensor(
            x, device = device
    ).float()
    basis_weight = model.model.compute_weight(torch_x)
    series_weights = (basis_weight[0] * model.model.output_linear.weight.transpose(1,0)).cpu().numpy()
    logits, _ = model(torch_x)
    logits = logits.cpu().numpy()


# #### In figure real value $X$ and $Y$ marked with circle, and predicted $\hat{Y}$ marked with x.

# In[7]:


fig, ax = plt.subplots(1,1)

st = sample_numb
ed = sample_numb + len(x[0])

ax.plot(range(st, ed), x[0], marker = 'o', markerfacecolor='none')
ax.set_xlabel('Time step')

for n, v in enumerate(np.concatenate([x[0, -1:], y]).transpose()):
    ax.plot(range(ed - 1, ed + 1), v, '--', color = list(MP.colors.TABLEAU_COLORS)[n])
    ax.plot(ed, v[1], marker = 'o', markerfacecolor='none', color = list(MP.colors.TABLEAU_COLORS)[n])
    ax.plot(ed, logits[n], marker = 'x', markerfacecolor='none', color = list(MP.colors.TABLEAU_COLORS)[n])

ax.set_rasterized(True)
ax.legend(log['columns'])


# In[8]:


plt.rcParams['font.size'] = 13

for n, series_name in enumerate(log['columns']):

    invert_sw = series_weights[:,n].reshape(-1,config.input_feature)
    


    max_v = invert_sw.max()
    min_v = invert_sw.min()
    thr_v = max_v if max_v > np.abs(min_v) else np.abs(min_v)

    fig, ax = plt.subplots(1,1,figsize = (7,5))

    sbn.heatmap(
        invert_sw.transpose(),
        vmin = -thr_v,
        vmax = thr_v,
        xticklabels = range(st, ed),
        yticklabels = log['columns'],
        cmap = 'coolwarm',
        ax = ax,
        annot = True, 
        fmt = '.2f'
    )

    ax.set_title(f'Contribution map for {series_name}')
    fig.tight_layout(pad=1.01)

