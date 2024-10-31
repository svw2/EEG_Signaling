import torch
import numpy as np
import matplotlib.pyplot as plt
class EEGDataset:
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
                tp9 = lst[789:789+512]
                tp10 = lst[789+512:789+2*512]
                self.images.append(np.array(self.turnfloat(image)))
                tp9 = np.array(self.turnfloat(tp9))[np.newaxis]
                tp10 = np.array(self.turnfloat(tp10))[np.newaxis]
                self.eegs.append(np.concatenate([tp9,tp10],axis = 0))

    def __len__(self):
        return len(self.eegs)
    
    def turnfloat(self, input):
        for i in range(len(input)):
            input[i]=float(input[i])
        return input

    def __getitem__(self, index):
        image = self.images[index]
        eeg = self.eegs[index]
        image = torch.tensor(image).float()/255
        eeg = torch.tensor(eeg).float()[None]
        return eeg, image
    
    def view(self,index):
        image = self.images[index].reshape(28,28,1)
        eeg = self.eegs[index]
        fig,axs = plt.subplots(1,2)
        axs[0].imshow(image,cmap="gray")
        axs[0].axis("off")
        axs[1].plot(eeg[0],"r-")
        axs[1].plot(eeg[1],"b-")
        plt.suptitle("Image and EEG Signal for #" + str(index))
        plt.show()

        

        


    

                    
                    

                    
                        

