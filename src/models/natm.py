from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Model
from .featurenn import FeatureNN
import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE



class natm_independent(nn.Module):
    def __init__(self,config):
        super().__init__()
        self._config = config
        
        
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
    
    
    

class natm_time(nn.Module):
    def __init__(self,config):
        super().__init__()
        self._config = config
        
        
        self.FN_dicts = nn.ModuleDict(
            {
                f'{i}' : FeatureNN(
                    config=config, 
                    name=f'{i}var', 
                    input_shape=1, 
                    num_units=config.nam_basis_functions, 
                    feature_num=i,
                    output_shape = 1,
                    mean=config.mean,std = config.std,
                    activ = config.nam_activation
                ) for i in range(config.input_feature)
        })
        
        self.output_linear = nn.Linear(config.input_feature * config.input_length,config.output_feature)
        self.output_dropout = nn.Dropout(self._config.nam_output_dropout)
        
    def feature_weight(self,index,x):
        return self.FN_dicts[index](x)
    
        
    def compute_weight(self,x):
        outputs = []
    
        for var  in range(self._config.input_feature):
            output = self.FN_dicts[str(var)](x[:,:,var])
            outputs.append(output)
        return torch.cat(outputs,1)
    
    
    def forward(self,x):
        weight = self.compute_weight(x)
        return self.output_linear(self.output_dropout(weight.squeeze())), weight
    
    
    
    
    

class natm_feature(nn.Module):
    def __init__(self,config):
        super().__init__()
        self._config = config
        
        self.FN_dicts = nn.ModuleDict(
            {
                f'{i}' : FeatureNN(
                    config=config, 
                    name=f'{i}time', 
                    input_shape=1, 
                    num_units=config.nam_basis_functions, 
                    feature_num=i,
                    output_shape = 1,
                    mean=config.mean,std = config.std,
                    activ = config.nam_activation
                ) for i in range(config.input_length)
        })
        
        self.output_linear = nn.Linear(config.input_feature * config.input_length,config.output_feature)
        self.output_dropout = nn.Dropout(self._config.nam_output_dropout)
        
        
    def feature_weight(self,index,x):
        return self.FN_dicts[index](x)
    
    
    def compute_weight(self,x):
        outputs = []
    
        for time in range(self._config.input_length):
            output = self.FN_dicts[str(time)](x[:,time,:])
            outputs.append(output)
        return torch.cat(outputs,1)

    
    def forward(self,x):
        weight = self.compute_weight(x)
        return self.output_linear(self.output_dropout(weight.squeeze())), weight
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


class pl_natm(pl.LightningModule): 
    def __init__(self, config, criterion = None, metric = None):
        super().__init__()
        self._config = config
        self.save_hyperparameters(ignore=['model'])
        
        '''
        NATMs (https://doi.org/10.1016/j.eswa.2023.120307)
        
        criterion (torch.optim) : Choose one of torch optimizer. 
                                  Customized optimizer is also possible if your optimizer held the manner of torch optimizer class.
                                  default = 'MSEloss'
                                  
        metric (pytorch_forecasting.metric) : You can use the metrics for regression problems. 
                                              I recommended you to use the metrics in pytorch_forecasting library.
                                              default = 'SMAPE'
        '''
        
        
        
        assert self._config.natm_type in ['Independent','Feature','Time'], \
        "Choose one of the types what our paper suggested as, ('Independent','Feature','Time')"
        
        if self._config.natm_type == 'Independent':
            self.model = natm_independent(self._config)
        elif self._config.natm_type == 'Time':
            self.model = natm_time(self._config)
        else:
            self.model = natm_feature(config)
        
        
        # loss
        self.criterion = nn.MSELoss()
        if criterion != None:
            self.criterion = criterion

            
        # metric
        self.metric = SMAPE()
        if metric != None:
            self.metric = metric
        
        
        
    def forward(self, x):
        x, sequence_weight = self.model(x)
        return x, sequence_weight

    
    def compute_loss(self,y_hat,y_true):
        loss = self.criterion(y_hat,y_true)
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
        logits, weights = self(x)  
        loss = self.compute_loss(logits, y) 
        metric = self.metric(logits, y)
        return y, logits, metric, weights
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._config.lr, weight_decay = self._config.l2_norm)
        return optimizer

    
