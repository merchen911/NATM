import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ExU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mean:float = 4.,
        std:float = .5
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters(mean,std)

    def reset_parameters(self,mean,std) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=mean, std=std)
        torch.nn.init.trunc_normal_(self.bias, std=std)

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
#         output = F.relu(output)
#         output = torch.clamp(output, 0, n)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
