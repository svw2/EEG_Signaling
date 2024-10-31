from eegdataset import *
from data2 import *
from nwk import *
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

tdata = EEGDataset("eegdata", "train")
vdata = EEGDataset("eegdata", "val")
batchsize = 3
tload = DataLoader(tdata, batch_size = batchsize, shuffle = True)
vload = DataLoader(vdata, batchsize)
error_function = nn.MSELoss()
nwk = NWK2((2,512))
proc = torch.optim.Adam(nwk.parameters(), lr = 0.002)
N = 50
min_error = float("inf")
for i in range(N):
    for image, eeg in tload:
        p = nwk(image)
        e = error_function(p,eeg[:,0])
        proc.zero_grad()
        e.backward()
        proc.step()
    print(f"index = {i}, error = {e}")
    nwk.eval()
    sum_e = 0
    for image, eeg in vload:
        with torch.no_grad():
            p = nwk(image)
            e = error_function(p,eeg)
            sum_e += e
    average_e = sum_e/len(vdata)
    print(f"index = {i}, error = {average_e}")
    nwk.train()
    if(average_e < min_error):
        min_error = average_e
        torch.save(nwk.state_dict(),"nwk.pth")
torch.save(nwk.state_dict(), "nwk_last.pth")

        



    