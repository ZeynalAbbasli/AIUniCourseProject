import torch 
import torch.nn
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
import math
import random
from pathlib import Path

class custom_dataset(Dataset):
    def __init__(self, mode = "train", root = './dataset/splitted/', transformsBasic = None, transformsAugment = None):
        super().__init__()
        self.mode = mode
        self.root = root
        self.transformsBasic = transformsBasic
        self.transformsAugment = transformsAugment
        
        #select split
        self.folder = os.path.join(self.root, self.mode)
        
        #initialize lists
        self.image_list = []
        self.label_list = []
        
        #save class lists
        self.class_list = os.listdir(self.folder)
        self.class_list.sort()
        random.seed(10)
        for class_id in range(len(self.class_list)):
            count = 0
            llst = []
            for image in os.listdir(os.path.join(self.folder, self.class_list[class_id])):
                count += 1
                self.image_list.append((os.path.join(self.folder, self.class_list[class_id], image), 'none'))
                llst.append((os.path.join(self.folder, self.class_list[class_id], image), 'none'))
                label = np.zeros(len(self.class_list))
                label[class_id] = 1.0
                self.label_list.append(label)
            if(count < 100 and self.mode == 'train'):
                numrand = random.randint(100, 120)
                extrasmpl = numrand - count
                if(extrasmpl <= count):
                    for i in range(extrasmpl):
                        self.image_list.append((llst[i][0],'augm'))
                        self.label_list.append(label)
                else:
                    k = math.ceil(extrasmpl/count)
                    for j in range(len(llst)):
                        for _ in range(k):
                            self.image_list.append((llst[j][0],'augm'))
                            self.label_list.append(label)

                
        
    def __getitem__(self, index):
        image_name = self.image_list[index][0]
        label = self.label_list[index]
        
        
        image = Image.open(image_name)
        if(self.transformsBasic):
            image = self.transformsBasic(image)
        if(self.mode == 'train' and self.transformsAugment and self.image_list[index][1] == 'augm'):
            image = self.transformsAugment(image)
        label = torch.tensor(label)
        
        return image, label
            
    def __len__(self):
        return len(self.image_list)        
