import os
import random
from statistics import mean
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import cv2
import scipy
from scipy.ndimage import morphology
import skimage.transform as skTrans
from dipy.align.imaffine import AffineMap
from dipy.align.imaffine import AffineRegistration
from dipy.align.transforms import AffineTransform3D
from dipy.align import (affine_registration, center_of_mass, translation, rigid, affine)
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from transformers import SamModel, SamConfig, SamProcessor
from segment_anything import sam_model_registry
from segment_anything.modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer)
import monai
from datasets import Dataset as _Dataset
import neptune
from tqdm import tqdm
#from model_C3_methods import *

affreg = AffineRegistration(level_iters=[0])
current_directory = os.environ['HOME']

# Load model and processor:
model = SamModel.from_pretrained("facebook/sam-vit-base") # Load the model
processor = SamProcessor.from_pretrained("facebook/sam-vit-base") # Initialize the processor
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
#print('Device:',device, ':',torch.cuda.device_count())


# Load data
test_data_path = os.path.join(current_directory, 'mri-infarct-segmentation/data/DUPONT/datasplit/FixedPath/test_data.csv')
df_test = pd.read_csv(test_data_path)
data = []
for i in range(0, len(df_test)):
    data.append([os.path.join(current_directory, df_test["DWI_path"][i]),
                os.path.join(current_directory, df_test["ADC_path"][i]),
                os.path.join(current_directory, df_test["b0"][i]),
                os.path.join(current_directory, df_test["Label_path"][i])])


# Functions for testing:
def get_bounding_box(input_vol, showPlot):
    bbox_vol = []
    for slice_idx in range(0,len(input_vol[0,2])):
        input_slice = input_vol[:,:,slice_idx]
        if np.sum(input_slice) > 0:
            y_indices, x_indices = np.where(input_slice > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = input_slice.shape

            # add perturbation to bounding box coordinates
            perturbation = 0 # 0 when testing
            x_min = max(0, x_min - perturbation)
            x_max = min(W, x_max + perturbation)
            y_min = max(0, y_min - perturbation)
            y_max = min(H, y_max + perturbation)
            bbox = [x_min, y_min, x_max, y_max]
            bbox_vol.append(bbox)

            if showPlot:
                # Create a subplot for displaying the image
                fig, axes = plt.subplots(1, 2, figsize=(6, 6))
                axes[0].imshow(input_slice, cmap='gray')
                axes[0].set_title('model segmentation')
                axes[0].axis('off')

                axes[1].imshow(input_slice, cmap='gray')
                axes[1].set_title('model segmentation with bboxes')
                axes[1].axis('off')
                box = bbox
                axes[1].plot([box[0], box[2]],[box[1], box[1]], color='green', linewidth=1)
                axes[1].plot([box[0], box[2]],[box[3], box[3]], color='green', linewidth=1)
                axes[1].plot([box[0], box[0]],[box[1], box[3]], color='green', linewidth=1)
                axes[1].plot([box[2], box[2]],[box[1], box[3]], color='green', linewidth=1)
                plt.show()
    return bbox_vol

def Create_DWI_symDif_map(DWI_vol):
    # Mirror the image horizontally
    DWI_vol_mir = np.flipud(DWI_vol)

    # Register DWI_vol to mirrored DWI
    affine3d = affreg.optimize(DWI_vol, DWI_vol_mir, AffineTransform3D(), params0=None) 

    # Transform the mirrored DWI to match the original DWI_vol
    DWI_vol_trans = affine3d.transform(DWI_vol_mir)

    # Left/Right difference for DWI_vol:
    DWI_symDif_vol = DWI_vol-DWI_vol_trans

    return DWI_symDif_vol

def Create3ChannelVolumes(DWI_vol, ADC_vol, b0_vol, symDif_vol):
    DWI_slice = DWI_vol
    DWI_slice = ((DWI_slice - np.min(DWI_slice)) / (np.max(DWI_slice) - np.min(DWI_slice))).astype(float)

    # Compute the symmetric difference of left and right hemisphere from DWI:
    Sym_dif_slice = symDif_vol
    Sym_dif_slice = ((Sym_dif_slice - np.min(Sym_dif_slice)) / (np.max(Sym_dif_slice) - np.min(Sym_dif_slice))).astype(float)
    Sym_dif_slice = np.where(((Sym_dif_slice*-1+1) < DWI_slice), DWI_slice, DWI_slice*0.5)

    # Compute the difference between the modalities ADC and b0:
    b0_th = np.where(DWI_slice > (np.mean(DWI_slice)), b0_vol, 0)
    ADC_th = np.where(DWI_slice > (np.mean(DWI_slice)), ADC_vol, 0)
    modalityDif_slice = np.where(b0_th > ADC_th , DWI_slice, DWI_slice * 0.5)
    
    # Compute a dif_dif slice based on the difference between the to dif_slices:
    dif_dif_map = Sym_dif_slice * modalityDif_slice

    # Stack slices into 3-channel slice:
    DWI_vol = np.expand_dims(DWI_vol, axis=0)
    ADC_vol = np.expand_dims(ADC_vol, axis=0)
    C3_vol = np.concatenate((DWI_vol, ADC_vol), axis=0)
    dif_dif_vol = np.asarray(dif_dif_map)
    dif_dif_vol = ((dif_dif_vol - dif_dif_vol.min()) / (dif_dif_vol.max() - dif_dif_vol.min()) * 255).astype(np.uint8)
    dif_dif_vol = np.expand_dims(dif_dif_vol, axis=0)
    C3_vol = np.concatenate((C3_vol, dif_dif_vol), axis=0)

    return C3_vol

def getVolume(DWI_path, ADC_path, b0_path, label_path, channels):
    label_vol = nib.load(label_path).get_fdata()
    while np.sum(label_vol) < 1:
        print('Error-- np.sum(label_vol) < 1 not....')
        idx = random.randint(0, len(data)-1)
        DWI_path, ADC_path, b0_path, label_path = data[idx]
        label_vol = nib.load(label_path).get_fdata()

    DWI_vol = nib.load(DWI_path).get_fdata()
    tmp = len(DWI_vol[0,0,:])
    vol_dim = (256, 256, tmp) #volumes are resized to the same dimention
    DWI_vol = skTrans.resize(DWI_vol, vol_dim, order=1, preserve_range=True)
    DWI_vol = ((DWI_vol - DWI_vol.min()) / (DWI_vol.max() - DWI_vol.min()) * 255).astype(np.uint8)

    if channels == "3C=DWI_ADC_DifDif":
        symDif_vol = Create_DWI_symDif_map(DWI_vol)
        symDif_vol = symDif_vol.transpose(2, 0, 1)
    else:
        symDif_vol = DWI_vol.transpose(2, 0, 1) #if symDif_vol is not needed, it is faster to set it to a dummy value
    DWI_vol = DWI_vol.transpose(2, 0, 1)
  
    ADC_vol = nib.load(ADC_path).get_fdata()
    ADC_vol = skTrans.resize(ADC_vol, vol_dim, order=1, preserve_range=True)
    ADC_vol = ((ADC_vol - ADC_vol.min()) / (ADC_vol.max() - ADC_vol.min()) * 255).astype(np.uint8)
    ADC_vol = ADC_vol.transpose(2, 0, 1)

    b0_vol = nib.load(b0_path).get_fdata()
    b0_vol = skTrans.resize(b0_vol, vol_dim, order=1, preserve_range=True)
    b0_vol = ((b0_vol - b0_vol.min()) / (b0_vol.max() - b0_vol.min()) * 255).astype(np.uint8)
    b0_vol = b0_vol.transpose(2, 0, 1)
    
    label_vol = nib.load(label_path).get_fdata()
    label_vol = skTrans.resize(label_vol, vol_dim, order=1, preserve_range=True)
    masks = label_vol.transpose(2, 0, 1)

    #Create a list to store the indices of non-empty masks
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
            
    # Filter the image and mask arrays to keep only the non-empty pairs
    DWI_vol = DWI_vol[valid_indices]
    ADC_vol = ADC_vol[valid_indices]
    b0_vol = b0_vol[valid_indices]
    symDif_vol = symDif_vol[valid_indices]
    filtered_masks = masks[valid_indices]

    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    dataset_dict = {"DWI_vol": [Image.fromarray(img) for img in DWI_vol],
                    "ADC_vol": [Image.fromarray(img) for img in ADC_vol],
                    "b0_vol": [Image.fromarray(img) for img in b0_vol],
                    "symDif_vol": [Image.fromarray(img) for img in symDif_vol],
                    "label": [Image.fromarray(mask) for mask in filtered_masks],
    }

    # Create the dataset using the datasets.Dataset class
    return _Dataset.from_dict(dataset_dict)

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor, type, prompt, channels):
    self.dataset = dataset
    self.processor = processor
    self.type = type
    self.prompt = prompt
    self.channels = channels

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    DWI_vol = item["DWI_vol"]
    ADC_vol = item["ADC_vol"]
    b0_vol = item["b0_vol"]
    symDif_vol = item["symDif_vol"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = self.prompt

    DWI_vol = np.array(DWI_vol)
    ADC_vol = np.array(ADC_vol)
    b0_vol = np.array(b0_vol)

    # Stack slices into 3-channel slice:
    if self.channels == "1C=DWI":
        DWI_vol = np.expand_dims(DWI_vol, axis=0)
        image = np.repeat(DWI_vol, 3, axis=0)
    if self.channels == "3C=DWI_ADC_b0":
        DWI_vol = np.expand_dims(DWI_vol, axis=0)
        ADC_vol = np.expand_dims(ADC_vol, axis=0)
        b0_vol = np.expand_dims(b0_vol, axis=0)
        image = np.concatenate((DWI_vol, ADC_vol), axis=0)
        image = np.concatenate((image, b0_vol), axis=0)
    if self.channels == "3C=DWI_ADC_DifDif":
        image = Create3ChannelVolumes(DWI_vol, ADC_vol, b0_vol, symDif_vol)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

def Find_DICE_vol(segmentation_mask, label_mask):
    segmentation_mask = segmentation_mask > 0.5 # binary
    areaOfOverlap = np.sum(segmentation_mask * label_mask).item()
    totalArea = np.sum(segmentation_mask).item() + np.sum(label_mask).item()
    dice_score = (2*areaOfOverlap)/totalArea
    return dice_score

def Find_DICE_slice(segmentation_mask, label_mask):
    segmentation_mask = segmentation_mask > 0.5 # binary
    areaOfOverlap = np.sum(segmentation_mask * label_mask)#.item()
    totalArea = np.sum(segmentation_mask).item() + np.sum(label_mask)#.item()
    dice_score = (2*areaOfOverlap)/totalArea

    return dice_score

def plotFunc(DWI_slice, SAM_out, Label_slice, box):    
            fig, axes = plt.subplots(1, 3, figsize=(9, 9))  # Adjust figsize as needed
            axes[0].imshow(DWI_slice, cmap='gray')  # Assuming grayscale images
            axes[0].set_title('DWI slice')
            axes[0].axes.axis('off')
            axes[0].plot([box[0], box[2]],[box[1], box[1]], color='green', linewidth=1)
            axes[0].plot([box[0], box[2]],[box[3], box[3]], color='green', linewidth=1)
            axes[0].plot([box[0], box[0]],[box[1], box[3]], color='green', linewidth=1)
            axes[0].plot([box[2], box[2]],[box[1], box[3]], color='green', linewidth=1)
            
            axes[1].imshow(SAM_out, cmap='gray')  # Assuming grayscale images
            axes[1].set_title('SAM output')
            axes[1].axes.axis('off')
            axes[2].imshow(Label_slice, cmap='gray')  # Assuming grayscale images
            axes[2].set_title('Label')
            axes[2].axes.axis('off')
            plt.show()

def plotAndDice(Label_vol, DWI_path, prompt, outputs):
    DWI_vol = nib.load(DWI_path).get_fdata()
    dice_vol = []
    jj = 0
    for j in range(0,len(Label_vol[0,0,:])):
        Label_slice = Label_vol[:,:,j]
        if np.sum(Label_slice) > 0:
                DWI_slice = DWI_vol[:,:,j]
                SAM_out = outputs[jj]
                if SAM_out.ndim == 2:
                    SAM_out = SAM_out > 0.5
                else:
                    SAM_out = SAM_out[jj,:,:] > 0.5

                # Dice:
                SAM_dice = Find_DICE_slice(SAM_out, Label_slice)
                print('Dice for slice:', round(SAM_dice,3))
                dice_vol.append(SAM_dice)

                # Plot:
                plotFunc(DWI_slice, SAM_out, Label_slice, prompt[jj])

                jj = jj+1
    print(f'Mean dice for subject: {round(mean(dice_vol),3)}') 
    return dice_vol