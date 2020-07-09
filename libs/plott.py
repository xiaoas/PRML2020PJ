import visdom
import matplotlib.pyplot as plt
import torch
try:
    vis = visdom.Visdom()
except Exception as ex:
    print(ex)
def plot(*curves):
    if len(curves) == 1:
        vis.line(*curves)
    else:
        lines = torch.stack([torch.tensor(i) for i in curves], dim = 1)
        vis.line(lines)