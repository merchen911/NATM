import torch
# from torch import nn
import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE



class DNN(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self._config = config

        blocks = [torch.nn.Flatten()]
        self.dnn_hiddens = [self._config.input_feature * self._config.input_length] + self._config.dnn_hiddens + [self._config.output_feature]
        
        layers = self.dnn_hiddens[:-1]
        
        for i, o in zip(layers, layers[1:]):
            blocks += [self.build_block(i,o)]
        else:
            blocks += [torch.nn.Linear(*self.dnn_hiddens[-2:])]
            self.blocks = torch.nn.ModuleList(blocks)

            
    def build_block(self, i, o):
        return torch.nn.Sequential(
            torch.nn.Linear(i,o),
            torch.nn.BatchNorm1d(o),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(self._config.dropout)
        )
        
        
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x
    
    
    
class pl_DNN(pl.LightningModule): 
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.save_hyperparameters(ignore=['model'])
        self.metric = SMAPE()
        self.model = DNN(self._config)

        # loss
        self.criterion = torch.nn.MSELoss()  

        
    def forward(self, x):
        x = self.model(x)
        return x.squeeze()

    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x)
        loss = self.criterion(logits, y.squeeze()) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=self._config.prog_bar, logger=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch   
        logits = self(x)  
        loss = self.criterion(logits, y.squeeze()) 
        metric = self.metric(logits, y.squeeze())

        metrics = {'val_metric': metric, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    
    def validation_step_end(self, val_step_outputs):
        metric  = val_step_outputs['val_metric'].cpu()
        loss = val_step_outputs['val_loss'].cpu()

        self.log('val_metric',  metric, prog_bar=self._config.prog_bar)
        self.log('val_loss', loss, prog_bar=self._config.prog_bar)

        
    def test_step(self, batch, batch_idx):
        x, y = batch  
        logits = self(x)  
        loss = self.criterion(logits, y.squeeze()) 
        metric = self.metric(logits, y.squeeze())
        metrics = {'test_metric': metric, 'test_loss': loss}
        self.log_dict(metrics)
        return metrics

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._config.lr)
        return optimizer
    
    
    def predict_step(self, batch, batch_idx):
        x, y = batch  
        logits = self(x)  
        loss = self.criterion(logits, y.squeeze()) 
        metric = self.metric(logits, y.squeeze())
        return y, logits, metric
    

