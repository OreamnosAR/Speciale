#!/usr/bin/env python

import os
import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as T
import nibabel as nib
import nibabel.processing
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import csv
import skimage.transform as skTrans
import neptune
import monai

current_directory = os.environ['HOME']

class CustomDataset(Dataset):
    def __init__(self, datapath):
        df_train=pd.read_csv(datapath)
        self.data = [] 
        for i in range(0, len(df_train)):
            self.data.append([(current_directory + '/' + df_train["DWI_path"][i]),
                              (current_directory + '/' + df_train["ADC_path"][i]),
                              (current_directory + '/' + df_train["b0"][i]),
                              (current_directory + '/' + df_train["Label_path"][i])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        DWI_path, ADC_path, b0_path, Label_path = self.data[idx]

        ADC_vol = nib.load(ADC_path).get_fdata()
        ADC_vol = ((ADC_vol - ADC_vol.min()) / (ADC_vol.max() - ADC_vol.min()) * 255).astype(np.uint8)
        ADC_tensor = self.resize_and_to_tensor(ADC_vol)

        DWI_vol = nib.load(DWI_path).get_fdata()
        DWI_vol = ((DWI_vol - DWI_vol.min()) / (DWI_vol.max() - DWI_vol.min()) * 255).astype(np.uint8)
        DWI_tensor = self.resize_and_to_tensor(DWI_vol)

        b0_vol = nib.load(b0_path).get_fdata()
        b0_vol = ((b0_vol - b0_vol.min()) / (b0_vol.max() - b0_vol.min()) * 255).astype(np.uint8)
        b0_tensor = self.resize_and_to_tensor(b0_vol)

        label_vol = nib.load(Label_path).get_fdata()
        label_tensor = self.resize_and_to_tensor(label_vol)

        scans_tensor = torch.cat([DWI_tensor, ADC_tensor, b0_tensor],0)
        return scans_tensor, label_tensor
    
    def resize_and_to_tensor(self, vol):
        #crop method: vol = vol[0:128, 0:128, 0:16]
        vol_dim = (128, 128, 32) #volumes are resized to the same dimention
        vol = skTrans.resize(vol, vol_dim, order=1, preserve_range=True)
        vol_tensor = torch.from_numpy(vol).float() #Convert to tensor
        vol_tensor = vol_tensor.unsqueeze(0) #Add channel dim

        return vol_tensor

traindata_path = os.path.join(current_directory,"mri-infarct-segmentation/data/General_dataset/datasplit/train_data.csv")
valdata_path = os.path.join(current_directory,"mri-infarct-segmentation/data/General_dataset/datasplit/val_data.csv")
trainDataset = CustomDataset(traindata_path)
valDataset = CustomDataset(valdata_path)

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv3d(3, 64, kernel_size=(3,3,3), padding=1) #
        self.e12 = nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=1) #
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=2) #

        self.e21 = nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=1) #
        self.e22 = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=1) #
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=2) #

        self.e31 = nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=1) #
        self.e32 = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=1) #
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=2) #

        self.e41 = nn.Conv3d(256, 512, kernel_size=(3,3,3), padding=1) #
        self.e42 = nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=1) #
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=2) #

        self.e51 = nn.Conv3d(512, 1024, kernel_size=(3,3,3), padding=1) #
        self.e52 = nn.Conv3d(1024, 1024, kernel_size=(3,3,3), padding=1) #


        # Decoder
        self.upconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=(2,2,2), stride=2) #
        self.d11 = nn.Conv3d(1024, 512, kernel_size=(3,3,3), padding=1) #
        self.d12 = nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=1) #

        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=(2,2,2), stride=2) #
        self.d21 = nn.Conv3d(512, 256, kernel_size=(3,3,3), padding=1)
        self.d22 = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=1)

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=(2,2,2), stride=2)
        self.d31 = nn.Conv3d(256, 128, kernel_size=(3,3,3), padding=1)
        self.d32 = nn.Conv3d(128, 128, kernel_size=(3,3,3), padding=1)

        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=(2,2,2), stride=2)
        self.d41 = nn.Conv3d(128, 64, kernel_size=(3,3,3), padding=1)
        self.d42 = nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=1)

        # Output layer
        self.outconv = nn.Conv3d(64, n_class, kernel_size=(1,1,1))

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        
        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        
        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)
        
        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)
        
        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))
        
        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))
        
        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))
        
        # Output layer
        out = self.outconv(xd42)
        out = torch.sigmoid(out)
        
        return out


nEpoch = 1000 
learningRate = 3e-06
batchSize = 2
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
model = UNet(n_class=1).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print('Device:',device, ':',torch.cuda.device_count())
loss_function = monai.losses.DiceLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

trainDataLoader = torch.utils.data.DataLoader(trainDataset,
                                              batch_size=batchSize,
                                              shuffle=True,
                                              num_workers = 0
                                              )
valDataLoader = torch.utils.data.DataLoader(valDataset,
                                            batch_size=batchSize,
                                            shuffle=True,
                                            num_workers = 0
                                            )


print('Training started')
#neptune:
run = neptune.init_run(
    project="MRI-infarct-segmentation/JAG",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkODc2ZGIzZC0wMjllLTQ2OGEtODlmZC05MGNmZDg2YzYwMWUifQ==",
)
params = {"Data": "JAG", "Model": "UNet3D_C3","Learning_rate": learningRate, "Optimizer": "Adam", "Lossfunction": "DiceLoss",}
run["parameters"] = params

train_loss_list = []
val_loss_list = []
best_val_loss = 99
_iter = 0
for iEpoch in range(nEpoch):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    total_train_loss = 0
    total_val_loss = 0

    print("Loading data for xbatch...")  
    for xbatch,ybatch in trainDataLoader:
        #print("Loading model output for xbatch...")  
        with torch.cuda.amp.autocast():
            output = model(xbatch.to(device))
        torch.cuda.empty_cache()

        #print("Loading model loss") 
        torch.cuda.empty_cache()
        with torch.cuda.amp.autocast():
            loss = loss_function(output, ybatch.to(device))
        
        #print("Loading updating model weights...") 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        total_train_loss += train_loss
        run["Training/train_loss"].append(loss) #neptune

        if _iter % 40 == 0:
            print('________________________ Iteration:', _iter,'________________________')
            print("Loss: " + str(train_loss))
            print()
        _iter+=1
        
        optimizer.zero_grad() #reset gradients
        torch.cuda.empty_cache()  # free unused GPU memory

    #Validation:
    print("Loading validation data...") 
    model.eval()
    for x_val, yval in valDataLoader:
        with torch.cuda.amp.autocast():
            val_output = model(x_val.to(device))
        torch.cuda.empty_cache()

        with torch.cuda.amp.autocast():
            val_loss = loss_function(val_output, yval.to(device))

        _val_loss = val_loss.item()
        total_val_loss += _val_loss
        run["Training/val_loss"].append(val_loss) #neptune

        if _val_loss < best_val_loss:
            best_val_loss = _val_loss
            run["Training/best_val_loss"].append(best_val_loss)
            torch.save(model.state_dict(), '/home/rosengaard/mri-infarct-segmentation/Anders/V3/Unet_C3/JAG/weights/Unet_C3_model.pt')
            print('New model saved with best val loss: ', best_val_loss)
        torch.cuda.empty_cache()
    
    print("Epoch complete") 
    train_loss_avg = total_train_loss / len(trainDataLoader)
    train_loss_list.append(train_loss_avg)
    print('Epoch training loss:', train_loss_avg)
    run["Training/epoch_train_loss"].append(train_loss_avg) #neptune

    val_loss_avg = total_val_loss / len(valDataLoader)
    val_loss_list.append(val_loss_avg)
    print('Epoch validation loss:', val_loss_avg)
    run["Training/epoch_val_loss"].append(val_loss_avg) #neptune
run.stop() #neptune
print('______________________________________________')
print('Training ended')