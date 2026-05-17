import torch 
import torch.nn as nn

def Neg_Sharpe(portfolio):
    return -torch.mean(portfolio) / torch.std(portfolio)


class SharpeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs_prev, future_rets):
        portflio = outputs_prev * future_rets 
        loss = Neg_Sharpe(portflio)
        return loss