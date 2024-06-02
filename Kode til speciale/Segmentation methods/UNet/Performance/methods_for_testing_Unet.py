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
from tqdm import tqdm
from statistics import mean

current_directory = os.environ['HOME']

class UNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv3d(n_channels, 64, kernel_size=(3,3,3), padding=1) #
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

class CustomDataset(Dataset):
    def __init__(self, datapath, channels, trained_on_dataset, type):
        self.channels = channels
        self.type = type
        if trained_on_dataset == "JAS":
            self.vol_dim = (256, 256, 32)
        if trained_on_dataset == "JAG":
            self.vol_dim = (128, 128, 32)

        self.data = [] 
        df_test = pd.read_csv(datapath) 
        for i in range(0, len(df_test)):
            self.data.append([(current_directory + '/' + df_test["DWI_path"][i]),
                              (current_directory + '/' + df_test["ADC_path"][i]),
                              (current_directory + '/' + df_test["b0"][i]),
                              (current_directory + '/' + df_test["Label_path"][i])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        DWI_path, ADC_path, b0_path, Label_path = self.data[idx]

        DWI_vol = nib.load(DWI_path).get_fdata()
        DWI_tensor = self.resize_and_to_tensor(DWI_vol)

        label_vol = nib.load(Label_path).get_fdata()
        if self.type == "train":
            label_tensor = self.resize_and_to_tensor(label_vol)
        if self.type == "test":
            label_tensor = torch.from_numpy(label_vol).float()
            label_tensor = label_tensor.unsqueeze(0)
        
        if self.channels == 1:
            return DWI_tensor, label_tensor

        if self.channels == 3:
            ADC_vol = nib.load(ADC_path).get_fdata()
            ADC_tensor = self.resize_and_to_tensor(ADC_vol)

            b0_vol = nib.load(b0_path).get_fdata()
            b0_tensor = self.resize_and_to_tensor(b0_vol)

            C3_tensor = torch.cat([DWI_tensor, ADC_tensor, b0_tensor],0)
            return C3_tensor, label_tensor
    
    def resize_and_to_tensor(self, vol):
        vol = ((vol - vol.min()) / (vol.max() - vol.min()) * 255).astype(np.uint8)
        
        # Padding: unable to run on Genome...
        #target_shape = (256, 256, 32) #volumes are resized to the same dimention
        #vol = np.pad(vol, ((0, 0), (0, 0), (0, target_shape[2] - vol.shape[2])), mode='constant')

        # Crop method: Removes more than 30% of the slices...
        #vol = vol[0:128, 0:128, 0:16]

        # Resize volumes to a smaller dimention:
        vol = skTrans.resize(vol, self.vol_dim, order=1, preserve_range=True)    

        # Convert to tensor:
        vol_tensor = torch.from_numpy(vol).float()
        vol_tensor = vol_tensor.unsqueeze(0) #Add channel dim

        return vol_tensor

def Find_DICE_slice(segmentation_mask, label_mask):
    if np.sum(label_mask) == 0:
         dice_score = 0
    else:
        areaOfOverlap = np.sum(segmentation_mask * label_mask)
        totalArea = np.sum(segmentation_mask).item() + np.sum(label_mask)
        dice_score = (2*areaOfOverlap)/totalArea

    return dice_score

def Find_DICE_vol(segmentation_mask, label_mask):
    segmentation_mask = segmentation_mask > 0.5 # binary
    areaOfOverlap = np.sum(segmentation_mask * label_mask).item()
    totalArea = np.sum(segmentation_mask).item() + np.sum(label_mask).item()
    dice_score = (2*areaOfOverlap)/totalArea
    return dice_score

def plotFunc(DWI_slice, output, Label_slice):    
            fig, axes = plt.subplots(1, 3, figsize=(7, 7))  # Adjust figsize as needed
            axes[0].imshow(DWI_slice, cmap='gray')  # Assuming grayscale images
            axes[0].set_title('DWI slice')
            axes[0].axes.axis('off')
            axes[1].imshow(output, cmap='gray')  # Assuming grayscale images
            axes[1].set_title('Unet output')
            axes[1].axes.axis('off')
            axes[2].imshow(Label_slice, cmap='gray')  # Assuming grayscale images
            axes[2].set_title('Label')
            axes[2].axes.axis('off')
            plt.show()

def plotAndDice(Label_vol, DWI_vol, outputs, trained_on_dataset):
    Label_vol = Label_vol.numpy()
    DWI_vol = DWI_vol.numpy()
    outputs = np.transpose(outputs.cpu().detach().numpy(), (0, 1, 2))

    # Model outputs are resized to match the label volume:
    vol_dim = Label_vol.shape
    outputs = skTrans.resize(outputs, vol_dim, order=1, preserve_range=True)

    #Dice for volume:
    dice_vol = Find_DICE_vol(outputs, Label_vol)

    dice_slices = []
    for j in range(0,len(Label_vol[0,0,:])):
        Label_slice = Label_vol[:,:,j] > 0.5
        output = outputs[:,:,j] > 0.5

        # Dice:
        if np.sum(output) > 0 or np.sum(output) > 0: # "Empty" slices are not interesting
            dice_slice = Find_DICE_slice(output, Label_slice)
            dice_slices.append(dice_slice)
            print('Dice for slice:', round(dice_slice,3))

        # Plot:
        if np.sum(Label_slice) > 0 or np.sum(output) > 0:     
                DWI_slice = DWI_vol[:,:,j]
                if trained_on_dataset == "JAG":
                    plotFunc(DWI_slice, output, Label_slice)
                if trained_on_dataset == "JAS": # Needs extra condition othervise there will be too many plots  
                    if np.sum(Label_slice) > 0: 
                        plotFunc(DWI_slice, output, Label_slice)

    return dice_slices, dice_vol