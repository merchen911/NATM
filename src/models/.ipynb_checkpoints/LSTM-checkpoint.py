import torch
# from torch import nn
import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE

class LSTM(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self._config = config

        self.lstm = torch.nn.LSTM(
            config.input_feature,
            hidden_size = config.lstm_hidden, num_layers = config.lstm_num_layers, 
            batch_first = True,
            bidirectional = config.lstm_bidirectional
        )
        if config.lstm_bidirectional:
            self.output_linear = torch.nn.Linear(2 * config.lstm_hidden,config.output_feature)
        else:
            self.output_linear = torch.nn.Linear(config.lstm_hidden,config.output_feature)
    
    def forward(self, inputs):
        x,(h,c) = self.lstm(inputs)
        return self.output_linear(x.mean(1))
    
    
    
class pl_LSTM(pl.LightningModule): 
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.save_hyperparameters(ignore=['model'])
        self.metric = SMAPE()
        self.model = LSTM(self._config)

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
    
    