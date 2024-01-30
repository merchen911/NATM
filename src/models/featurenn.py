import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Model

from .activation import ExU
from .activation import LinReLU


class FeatureNN(Model):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        config,
        name,
        *,
        input_shape: int,
        output_shape: int,
        num_units: int,
        mean = 4.,
        std = .5,
        feature_num: int = 0,
        activ = 'relu', 
#         nam_output_bias = True
    ) -> None:
        """Initializes FeatureNN hyperparameters.

        Args:
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          feature_num: Feature Index used for naming the hidden layers.
        """
        super(FeatureNN, self).__init__(config, name)
        self._input_shape = input_shape
        self._num_units = num_units
        self._feature_num = feature_num
        self.dropout = nn.Dropout(p=self.config.dropout)
        self._output_shape = output_shape
        self._activ = activ

        layers = []
        
        if self.config.nam_hidden_sizes != None:
            hidden_sizes = [self._num_units] + self.config.nam_hidden_sizes

            if self.config.nam_activation == "exu":
                layers.append(ExU(in_features=input_shape, out_features=num_units, mean = mean, std = std))
            else:
                layers.append(LinReLU(in_features=input_shape, out_features=num_units))

            ## Hidden Layers
            for in_features, out_features in zip(hidden_sizes, hidden_sizes[1:]):
                layers.append(LinReLU(in_features, out_features)) # NAM default
#                 layers.append(F.relu(nn.Linear(in_features, out_features)))
#                 layers.append(F.leaky_relu(nn.Linear(in_features, out_features)))
            ## Last Linear Layer
            layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=self._output_shape, bias = config.nam_output_bias))

        else:
            if self.config.nam_activation == "exu":
                layers.append(ExU(in_features=input_shape, out_features=num_units, mean = mean, std = std))
            else:
                layers.append(LinReLU(in_features=input_shape, out_features=num_units))
            ## Last Linear Layer
            layers.append(nn.Linear(in_features=num_units, out_features=self._output_shape, bias = config.nam_output_bias))
            


        self.model = nn.ModuleList(layers)
        # self.apply(init_weights)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        outputs = inputs.unsqueeze(-1)
        for layer in self.model[:-1]:
            outputs = layer(outputs)
            if self._activ == 'relu':
                outputs = F.relu(outputs)
            else:
                outputs = F.leaky_relu(outputs,.1)
            outputs = self.dropout(outputs)
        else:
            return self.model[-1](outputs)
        
        
        
def extractor(config,side):
    assert side in ['feature','sequence'], "Choose one of [feature, sequence]"
    if side == 'feature':
        return nn.ModuleList([
            FeatureNN(config=config, 
                      name=f'FeatureNN_{i}', 
                      input_shape=1, 
                      num_units=config.nam_basis_functions, 
                      feature_num=i,
                      output_shape = len(config.output_feature),
                      mean=config.mean,std = config.std,
                      activ = config.activation
                     )
            for i in range(len(config.input_feature))
        ])
    elif side == 'sequence':
        return nn.ModuleList([
            FeatureNN(config=config, 
                      name=f'SequenceNN_{i}', 
                      input_shape=1, 
                      num_units=config.nam_basis_functions, 
                      feature_num=i,
                      output_shape = config.output_length,
                      mean=config.mean,std = config.std,
                      activ = config.activation
                     )
            for i in range(config.input_length)
        ])