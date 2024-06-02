import os
from statistics import mean
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import monai
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader
import scipy
from scipy.ndimage import morphology
from dipy.align import (affine_registration, center_of_mass, translation, rigid, affine)
from dipy.align.imaffine import AffineMap
from dipy.align.transforms import AffineTransform3D
from dipy.align.imaffine import AffineRegistration
import skimage.transform as skTrans
import neptune

current_directory = os.environ['HOME']
dim = 23 # Dimention of scan along z-axis
input_size = dim*128 # With dim 23 size is 23*128=2944   - 128 is half of 256 cuz of DS = 2
output_size = dim*6 # 23*6 = 138

class CustomModel(nn.Module):
  def __init__(self):#, n_features, encoding_dim):
    super(CustomModel, self).__init__()
    self.encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 1472),
        nn.ReLU(),
        nn.Linear(1472, 736),
        nn.ReLU(),
        nn.Linear(736, 368),
        nn.ReLU(),
        nn.Linear(368, 184),
        nn.ReLU(),
        nn.Linear(184, 92),
    )

    self.decoder = nn.Sequential(
        nn.ReLU(),
        nn.Linear(92,115),
        nn.ReLU(),
        nn.Linear(115,138),
        nn.ReLU(),
        nn.Linear(138,output_size),
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class CustomDataset(Dataset):
    def __init__(self, data, type):
        self.type = type
        self.data = [] 
        for i in range(len(data)):
            self.data.append(data[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.type == "train":
            DWI_path, ADC_path, b0_path, Label_path, symDif_path = self.data[idx]
            symDif_vol = nib.load(symDif_path).get_fdata()
        if self.type == "test":
            DWI_path, ADC_path, b0_path, Label_path = self.data[idx]
            symDif_vol = Create_DWI_symDif_map(nib.load(DWI_path).get_fdata())

        DWI_vol = nib.load(DWI_path).get_fdata()
        DWI_tensor = self.resize_and_to_tensor(DWI_vol, True)
        
        ADC_vol = nib.load(ADC_path).get_fdata()
        ADC_tensor = self.resize_and_to_tensor(ADC_vol, True)

        b0_vol = nib.load(b0_path).get_fdata()
        b0_tensor = self.resize_and_to_tensor(b0_vol, True)

        # Compute the symDif_vol: symmetric difference of left and right hemisphere from DWI:
        symDif_tensor = self.resize_and_to_tensor(symDif_vol, True)

        label_vol = nib.load(Label_path).get_fdata()
        label_tensor = self.resize_and_to_tensor(label_vol, False)

        return DWI_tensor, ADC_tensor, b0_tensor,label_tensor, symDif_tensor
    
    def resize_and_to_tensor(self, vol, norm):
        if norm:
            vol = ((vol - vol.min()) / (vol.max() - vol.min()) * 255).astype(np.uint8)

        #Padding:
        target_shape = (256, 256, dim) #volumes are resized to the same dimention
        vol = np.pad(vol, ((0, 0), (0, 0), (0, target_shape[2] - vol.shape[2])), mode='constant')

        vol_tensor = torch.from_numpy(vol).float() #Convert to tensor
        return vol_tensor

def preprocessInput(histograms):
    inps = torch.tensor(histograms).float()
    return inps

def preprocessY(inps):
    inps = np.asarray(inps)
    inps = torch.tensor(inps).float()
    return inps

def get_histo(volume):
    volume = volume.numpy()
    myTH = np.mean(volume)*2
    myTH2 = 255

    DS = 2
    _w = int(256/DS)
    histograms = []
    for i in range(volume.shape[2]):
        histogram, bins = np.histogram(volume[:,:,i].flatten(), bins=_w, range=(myTH, myTH2))
        histograms.append(histogram)

    histograms_tensor = torch.tensor(histograms).float()
    return histograms_tensor.unsqueeze(0) #Add sample dim -> sample × channels × rows × columns

def DifDifSeg(DWI_symDif_vol, DWI_vol, ADC_vol, b0_vol, ggg):
    # Compute the average pixel values for DWI volume:
    avg_pixel_value_DWI_vol = np.mean(DWI_vol)
    avg_pixel_value_DWI_symDif_vol = np.mean(DWI_symDif_vol)

    _dif_dif_maps = []
    _modalityDif_vol = []
    _sym_dif_vol = []

    for slice_idx in range(0,len(DWI_vol[0,2])):
        g1, g2, g3, g4, g5, g6 = ggg[slice_idx]

        DWI_slice = DWI_vol[:,:,slice_idx]
        DWI_slice = ((DWI_slice - np.min(DWI_slice)) / (np.max(DWI_slice) - np.min(DWI_slice))).astype(float)

        # Reduce pepper noise artifacts in the created symDif volume:
        DWI_symDif_vol = np.where((DWI_symDif_vol > avg_pixel_value_DWI_vol * g1), DWI_symDif_vol, avg_pixel_value_DWI_symDif_vol)
    
        # Compute the symmetric difference of left and right hemisphere from DWI:
        Sym_dif_slice = DWI_symDif_vol[:,:,slice_idx]
        Sym_dif_slice = ((Sym_dif_slice - np.min(Sym_dif_slice)) / (np.max(Sym_dif_slice) - np.min(Sym_dif_slice))).astype(float)
        Sym_dif_slice = np.where(((Sym_dif_slice*-1+1) < DWI_slice), DWI_slice, DWI_slice*0.5)
        _sym_dif_vol.append(Sym_dif_slice)
        
        # Compute the modDif-slice: difference between the modalities ADC and b0:
        b0_th = np.where(DWI_slice > (np.mean(DWI_slice)*g2), b0_vol[:,:,slice_idx], 0)
        ADC_th = np.where(DWI_slice > (np.mean(DWI_slice)*g3), ADC_vol[:,:,slice_idx], 0)
        modalityDif_slice = np.where(b0_th > ADC_th, DWI_slice, DWI_slice * 0.5)
        _modalityDif_vol.append(modalityDif_slice)
        
        # Compute the DifDif-slice based on the symDif and modDif-slice:
        dif_dif_map = (Sym_dif_slice * modalityDif_slice)

        # Enhance highlighted areas:
        dif_dif_map = np.where(dif_dif_map > np.mean(dif_dif_map)*g4, dif_dif_map, dif_dif_map * 0.5)
        dif_dif_map = dif_dif_map * dif_dif_map
        dif_dif_map = ((dif_dif_map - np.min(dif_dif_map)) / (np.max(dif_dif_map) - np.min(dif_dif_map))).astype(float)
        
        # Remove salt noise from DifDif-slice:
        ADC_slice = ADC_vol[:,:,slice_idx]
        ADC_slice = ((ADC_slice - np.min(ADC_slice)) / (np.max(ADC_slice) - np.min(ADC_slice))).astype(float)
        ADC_slice = np.where((ADC_slice > 0.1), ADC_slice*-1, 0)
        tmp = (DWI_slice * ADC_slice)
        tmp = ((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))).astype(float)
        tmp = np.where((tmp > DWI_slice*g5), tmp, tmp*0.5)
        dif_dif_map = dif_dif_map * tmp

        # Normalize and binarize DifDif-slice:
        dif_dif_map = ((dif_dif_map - np.min(dif_dif_map)) / (np.max(dif_dif_map) - np.min(dif_dif_map))).astype(float)
        dif_dif_map = dif_dif_map > 0.5

        # Binary morhpology operations to further remove noise:
        dif_dif_map = scipy.ndimage.binary_closing(dif_dif_map, structure=np.ones((4,4))).astype(int)
        dif_dif_map = scipy.ndimage.binary_opening(dif_dif_map, structure=np.ones((2,2))).astype(int)
        dif_dif_map = scipy.ndimage.binary_dilation(dif_dif_map, structure=np.ones((5,5))).astype(int)

        # Add information from hyper intense areas if the DWI-slice back to the DifDif-slice:
        DWI_slice_TH = DWI_slice > g6
        dif_dif_map = dif_dif_map & DWI_slice_TH
   
        _dif_dif_maps.append(dif_dif_map)

    return _sym_dif_vol, _modalityDif_vol, _dif_dif_maps

def segVol(DWI_vol, ADC_vol, b0_vol, DWI_vol_sym_dif_map, _ggg):
    result = []
    _ggg = _ggg[0][:]
    gggs = []
    for i in range (len(DWI_vol[0,0,:])):
        ggg = _ggg[i]
        g1 = float(ggg[0].item())
        g2 = float(ggg[1].item())
        g3 = float(ggg[2].item())
        g4 = float(ggg[3].item())
        g5 = float(ggg[4].item())
        g6 = float(ggg[5].item())
        gggs.append([g1, g2, g3, g4, g5, g6])

    dif_dif_maps_new, _Sym_dif_slice_TH, _modalityDif_vol = DifDifSeg(DWI_vol_sym_dif_map.numpy(), DWI_vol.numpy(), ADC_vol.numpy(), b0_vol.numpy(), gggs)
    result.append(np.transpose(np.asarray(dif_dif_maps_new), (1, 2, 0)))
    result.append(np.transpose(np.asarray(_Sym_dif_slice_TH), (1, 2, 0)))
    result.append(np.transpose(np.asarray(_modalityDif_vol), (1, 2, 0)))
     
    return result

def Create_DWI_symDif_map(DWI_vol):
    affreg = AffineRegistration(level_iters=[0])
    # Mirror the image horizontally
    DWI_vol_mir = np.flipud(DWI_vol)

    # Register DWI_vol to mirrored DWI
    affine3d = affreg.optimize(DWI_vol, DWI_vol_mir, AffineTransform3D(), params0=None) 

    # Transform the mirrored DWI to match the original DWI_vol
    DWI_vol_trans = affine3d.transform(DWI_vol_mir)

    # Left/Right difference for DWI_vol:
    DWI_symDif_vol = DWI_vol-DWI_vol_trans

    return DWI_symDif_vol

def Find_DICE_slice(prediction, target):
    target_flat = target.view(-1)
    prediction_flat = prediction.view(-1)
    prediction_flat = prediction_flat > 0.5 # binary

    intersection = torch.sum(prediction_flat * target_flat)
    union = torch.sum(prediction_flat) + torch.sum(target_flat)

    dice = (2. * intersection) / (union)

    return dice.item()

def Find_DICE_vol(prediction, target):
    target_flat = target.view(-1)
    prediction_flat = prediction.view(-1)
    prediction_flat = prediction_flat > 0.5 # binary

    intersection = torch.sum(prediction_flat * target_flat)
    union = torch.sum(prediction_flat) + torch.sum(target_flat)

    dice = (2. * intersection) / (union)
    return dice.item()