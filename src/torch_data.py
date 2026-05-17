import torch 
from torch.utils import data
import numpy as np 

class MyDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, datax, datay):
        """Initialization""" 
        self.datax = torch.from_numpy(datax.astype(np.float32))
        self.datay = torch.from_numpy(datay.astype(np.float32))
        self.length = len(datax)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.datax[index], self.datay[index]