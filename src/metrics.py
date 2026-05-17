import numpy as np 

def report_metrics(ret):
    res = {}
    res['annual_ret'] = np.mean(ret) * 252
    res['annual_std'] = np.std(ret) * np.sqrt(252)
    res['annual_sharpe'] = (np.mean(ret) / np.std(ret)) * np.sqrt(252)
    return res