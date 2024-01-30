from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Model
from .featurenn import FeatureNN
import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE



class nam(nn.Module):
    def __init__(self,config):
        super().__init__()
        self._config = config
#         self._name = name
        
        
        self.FN_dicts = nn.ModuleDict(
            {
                f'{t}_{i}' : FeatureNN(
                    config=config, 
                    name=f'{t}time_{i}var', 
                    input_shape=1, 
                    num_units=config.nam_basis_functions, 
                    feature_num=i,
                    output_shape = 1,
                    mean=config.mean,std = config.std,
                    activ = config.nam_activation
                ) for t in range(config.input_length) for i in range(config.input_feature)
        })
        self.output_linear = nn.Linear(len(self.FN_dicts),config.output_feature)
        self.output_dropout = nn.Dropout(self._config.nam_output_dropout)
        
#         self._bias = torch.nn.Parameter(data=torch.zeros(len(config.output_feature)))
        
    def feature_weight(self,index,x):
        return self.FN_dicts[index](x)
        
        
        
    def compute_weight(self,x):
        outputs = []
        for keys, nn in self.FN_dicts.items():
            time, var = [int(i) for i in keys.split('_')]
            output = nn(x[:,time,var])
            outputs.append(output)
        return torch.stack(outputs,1)
    
    def forward(self,x):
        weight = self.compute_weight(x)
        return self.output_linear(self.output_dropout(weight.squeeze())), weight
#         return self.output_linear(weight.squeeze()), weight
#         return weight.sum(1) + self._bias, weight
    


class pl_nam(pl.LightningModule): 
    def __init__(self, 
                 config              
                 ):
        super().__init__()
        self._config = config
        self.save_hyperparameters(ignore=['model'])
        self.metric = SMAPE()

        self.model = nam(config)
        self.L2_len = len([i for i in self.model.parameters()])

        # loss
        self.criterion = nn.MSELoss()  

    def forward(self, x):
        x, sequence_weight = self.model(x)
        return x, sequence_weight

    def compute_loss(self,y_hat,y_true):
        loss = self.criterion(y_hat,y_true)
        
        if self._config.l2_regularization > 0:
            loss += sum(torch.linalg.norm(i, 2) for i in self.parameters()) * self._config.l2_regularization
            
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits, _ = self(x)
        loss = self.compute_loss(logits, y) 
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch   
        logits, _ = self(x)  
        loss = self.compute_loss(logits, y) 
        mape = self.metric(logits, y)

        metrics = {'val_SMAPE': mape, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_met  = val_step_outputs['val_SMAPE'].cpu()
        val_loss = val_step_outputs['val_loss'].cpu()

        self.log('val_SMAPE',  val_met, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        
    def test_step(self, batch, batch_idx):
        x, y = batch  
        logits, _ = self(x)  
        loss = self.compute_loss(logits, y) 
        metric = self.metric(logits, y)
        metrics = {'test_SMAPE': metric, 'test_loss': loss}
        self.log_dict(metrics)
        return metrics

    
    def predict_step(self, batch, batch_idx):
        x, y = batch  
        logits, _ = self(x)  
        loss = self.compute_loss(logits, y) 
        metric = self.metric(logits, y)
        return y, logits, metric
    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._config.lr, )
        return optimizer

    
    