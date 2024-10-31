from data2 import *
from nwk import *
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


data = EEG2(".", "test")
nwk = NWK(25600)
nwk.load_state_dict(torch.load("nwk.pth"))
p1 = nwk(data[0][0][None])
p2 = nwk(data[1][0][None])
p3 = nwk(data[2][0][None])
p4 = nwk(data[3][0][None])



