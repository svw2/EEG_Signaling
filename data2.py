import torch
import numpy as np
class EEG2:
    def __init__(self, path, dstype = "train"):
        self.path = path
        self.dstype = dstype
        self.images = []
        self.digits = []
        self.eegs = []
        with open(f"{path}/{dstype}.txt", "r") as f:
            for line in f:
                lst = line.strip("\n").split(",")
                self.digits.append(int(lst[2]))
                image = lst[3:787]
                eeg = lst[788:]
                for i in range (len(image)):
                    image[i] = float(image[i])
                self.images.append(np.array(image))
                for i in range(len(eeg)):
                    eeg[i] = float(eeg[i])
                self.eegs.append(np.array(eeg))
        self.eeg_length = self.eegs[0].shape[0]

    def __len__(self):
        return len(self.eegs)
    
    def __getitem__(self, index):
        image = self.images[index]
        eeg = self.eegs[index]
        image = torch.tensor(image).view(1,28,28).float()/255
        eeg = torch.tensor(eeg).float()[None]/300
        return image, eeg


    

                    
                    

                    
                        

