# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: PyTorch 1.11 (NGC 22.02/Python 3.8 Conda) on Backend.AI
#     language: python
#     name: python3
# ---

# !pip install tqdm
# !pip install wandb
# !pip install pytorch_pretrained_vit
# !pip install torchmetrics

# +
import wandb
import time

now = time
name = now.strftime('%Y-%m-%d %H:%M:%S')
pth_name = now.strftime('%Y%m%d%H%M%S')


# +
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchmetrics.functional.classification as metric
# %matplotlib inline
from torch import tensor
import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold

from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from skimage import io, transform

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
import tqdm

from pytorch_pretrained_vit import ViT


# -

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_df = pd.read_csv("train.csv", index_col=0)
test_df = pd.read_csv("test.csv", index_col=0)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize(size=(384,384)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_rand = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(384,384)),              
    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# +
class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.path = data["image"]
        self.label = data["label"]
        self.transform = transform
        



    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        image = torchvision.io.read_image("./train_crop/"+self.path.iloc[idx])
        image = transform_rand(image)
        #image = transform(image)

        
        label = torch.from_numpy(np.array(self.label.iloc[idx], dtype= np.float32))


        
        return image,label
    
class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.path = data["image"]
        self.label = data["label"]
        self.transform = transform
        



    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        image = torchvision.io.read_image("./test_crop/"+self.path.iloc[idx])
        image = transform(image)
        
        label = torch.from_numpy(np.array(self.label.iloc[idx], dtype= np.float32))

        
        return image,label
# -

train_data = Train_Dataset(train_df)
test_data = Test_Dataset(test_df)

train_loader = DataLoader(train_data, batch_size = 16, shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
test_loader = DataLoader(test_data, batch_size = 16,num_workers=12, shuffle=False, drop_last=True)

resnet152 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=False)
vgg19_bn = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet152
        num_ftrs = 2048
        self.resnet.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.3),
                                    nn.Linear(512, 2))

        
      
    def forward(self, x):
        return self.resnet(x)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = vgg19_bn
        num_ftrs = in_in_features
        self.vgg.classifier = in_features 
        
      
    def forward(self, x):
        return self.vgg(x)

vit =  ViT('B_16_imagenet1k', pretrained=True)
class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.vit = vit
        self.linear = nn.Linear(1000, 1)

        
      
    def forward(self, x):
        x = self.vit(x)
        x = torch.sigmoid(self.linear(x))
        return x

# +
if __name__ == "__main__":
    
    PATH = './weights/'
    parser = argparse.ArgumentParser(
        prog=".",
        description="..",
        epilog="..."
    )
    
    parser.add_argument("-m", "--model", choices=["vgg", "vit", "resnet"], help="pretrained model", required=True)
    #parser.add_argument("-lr", "--model", type=int, nargs=1, help="2개의 값을 입력하세요", required=True)
    args = parser.parse_args()
    
    
    
    wandb.init(entity="xodbs1270",  
        project="focus_check",
        name= str(args.model)+"_"+name)
    
    
    if args.model == "vgg":
        model = VGG().to(device)

        
    if args.model == "resnet":
        model = vit.to(device)
        

        
    if args.model == "vit":
        model = VisionTransformer().to(device)
    
    learning_rate = 0.0001
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-3)
    epochs = 30
    
    print("=============================")
    print("LR : %f Epoch : %d "%(learning_rate,epochs) )
    print("=============================")
    
    print(epochs)
    for epoch in range(epochs):
        
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm.tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output.squeeze(), label)
            #print(output)
        

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            scheduler.step(loss)

            #acc = (output.argmax(dim=1) == label).float().mean()
            #print(output.argmax(dim=1))
            #print(label)
            
            #tp, fp, tn, fn
            #train_conf_mat += metric.binary_stat_scores(label.to("cpu"), output.argmax(dim=1).to("cpu"))
            #print(train_conf_mat)
            #epoch_accuracy += acc / len(train_loader)
        
            epoch_loss += loss / len(train_loader)
            #print(output.squeeze())
            #print(label)
            #print(output)
        

        with torch.no_grad():
            #epoch_val_accuracy = 0
            epoch_val_loss = 0
            test_conf_mat = tensor([0,0,0,0,0])
            for data, label in tqdm.tqdm(test_loader):
                data = data.to(device)
                label = label.to(device)
            

                val_output = model(data)
                
                val_loss = criterion(val_output.squeeze(), label)
                
            

                #acc = (val_output.argmax(dim=1) == label).float().mean()
                #print(val_output.argmax(dim=1))
                
                
                
                #test_conf_mat += metric.binary_stat_scores(label.to("cpu"), val_output.argmax(dim=1).to("cpu"))
                #print(test_conf_mat)
                
                #epoch_val_accuracy += acc / len(test_loader)
                epoch_val_loss += val_loss / len(test_loader)
        #total_train_conf = train_conf_mat[0] + train_conf_mat[1] + train_conf_mat[2] + train_conf_mat[3]
        #total_test_conf = test_conf_mat[0] + test_conf_mat[1] + test_conf_mat[2] + test_conf_mat[3]

        """wandb.log({"train_loss" : epoch_loss,
                "train_acc": epoch_accuracy,
                "test_loss" : epoch_val_loss,
                "test_acc": epoch_val_accuracy,
                  "train_tp": train_conf_mat[0] / (train_conf_mat[0] + train_conf_mat[1]),
                  "train_tn": train_conf_mat[1]/ (train_conf_mat[0] + train_conf_mat[1]),
                  "train_fp": train_conf_mat[2]/ (train_conf_mat[2] + train_conf_mat[3]),
                   "train_fn": train_conf_mat[3]/ (train_conf_mat[2] + train_conf_mat[3]),
                    "test_tp": test_conf_mat[0] / (test_conf_mat[0] + test_conf_mat[1]),
                  "test_tn": test_conf_mat[1]/ (test_conf_mat[0] + test_conf_mat[1]),
                  "test_fp": test_conf_mat[2]/(test_conf_mat[2] + test_conf_mat[3]),
                   "test_fn": test_conf_mat[3]/ (test_conf_mat[2] + test_conf_mat[3])})"""
        
        wandb.log({"train_loss" : epoch_loss, "test_loss" : epoch_val_loss})
        
        #model.eval()
        #print(model(transform(torchvision.io.read_image("val_image/F_1.jpg"))))
        #print(model(transform(torchvision.io.read_image("val_image/F_2.jpg"))))
        #model.train()
        
        """print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )"""
        #torch.save(model, PATH + str(args.model)+"_"+name+ "_epoch" + str(epoch) + '_model.pt')  # 전체 모델 저장
        torch.save(model.state_dict(),PATH + str(args.model)+"_"+name + "_epoch" + str(epoch) + '_model.pt')  # 모델 객체의 state_dict 저장
        
    

    
