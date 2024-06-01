#!/usr/bin/env python

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib
import skimage.transform as skTrans #Resize image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import monai
from datasets import Dataset as _Dataset
from tqdm import tqdm
from statistics import mean
import neptune
from transformers import SamModel, SamConfig, SamProcessor
from dipy.align.imaffine import AffineRegistration
from dipy.align.transforms import AffineTransform3D
affreg = AffineRegistration(level_iters=[0])

#Load data
current_directory = os.environ['HOME']

#Train split
data_path = os.path.join(current_directory, 'mri-infarct-segmentation/Anders/V4/Slurm/datasplit_finetuning/train_data.csv')
data_df = pd.read_csv(data_path)
train_data = []
for i in range(0, len(data_df)):
    train_data.append([os.path.join(current_directory + '/' + data_df["DWI_path"][i]),
                os.path.join(current_directory + '/' + data_df["ADC_path"][i]),
                os.path.join(current_directory + '/' + data_df["b0"][i]),
                os.path.join(current_directory + '/' + data_df["Label_path"][i])])

#Validation split
data_path = os.path.join(current_directory, 'mri-infarct-segmentation/Anders/V4/Slurm/datasplit_finetuning/val_data.csv')
data_df = pd.read_csv(data_path)
val_data = []
for i in range(0, len(data_df)):
    val_data.append([os.path.join(current_directory, data_df["DWI_path"][i]),
                os.path.join(current_directory, data_df["ADC_path"][i]),
                os.path.join(current_directory, data_df["b0"][i]),
                os.path.join(current_directory, data_df["Label_path"][i])])

def Create_DWI_symDif_map(DWI_vol):# , slice_idx):
    # Mirror the image horizontally
    DWI_vol_mir = np.flipud(DWI_vol)

    # Register DWI_vol to mirrored DWI
    affine3d = affreg.optimize(DWI_vol, DWI_vol_mir, AffineTransform3D(), params0=None) 

    # Transform the mirrored DWI to match the original DWI_vol
    DWI_vol_trans = affine3d.transform(DWI_vol_mir)

    # Left/Right difference for DWI_vol:
    DWI_symDif_vol = DWI_vol-DWI_vol_trans

    return DWI_symDif_vol

# Method that creats the 3-channel maps:
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

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map, type):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = ground_truth_map.shape

    # Add perturbation to bounding box coordinates
    if type == "train":
        perturbation = np.random.randint(0, 3) 
    else:
        perturbation = 0
    x_min = max(0, x_min - perturbation)
    x_max = min(W, x_max + perturbation)
    y_min = max(0, y_min - perturbation)
    y_max = min(H, y_max + perturbation)
    bbox = [x_min, y_min, x_max, y_max]

    return bbox

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor, type):
    self.dataset = dataset
    self.processor = processor
    self.type = type

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
    prompt = get_bounding_box(ground_truth_mask, self.type)

    DWI_vol = np.array(DWI_vol)
    ADC_vol = np.array(ADC_vol)
    b0_vol = np.array(b0_vol)
    image = Create3ChannelVolumes(DWI_vol, ADC_vol, b0_vol, symDif_vol)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

def getVolume(idx):
    DWI_path, ADC_path, b0_path, label_path = data[idx]

    label_vol = nib.load(label_path).get_fdata()
    while np.sum(label_vol) < 1:
        idx = random.randint(0, len(data)-1)
        DWI_path, ADC_path, b0_path, label_path = data[idx]
        label_vol = nib.load(label_path).get_fdata()

    DWI_vol = nib.load(DWI_path).get_fdata()
    tmp = len(DWI_vol[0,0,:])
    vol_dim = (256, 256, tmp) #volumes are resized to the same dimention
    DWI_vol = skTrans.resize(DWI_vol, vol_dim, order=1, preserve_range=True)
    DWI_vol = ((DWI_vol - DWI_vol.min()) / (DWI_vol.max() - DWI_vol.min()) * 255).astype(np.uint8)

    symDif_vol = Create_DWI_symDif_map(DWI_vol)
    DWI_vol = DWI_vol.transpose(2, 0, 1)
    symDif_vol = symDif_vol.transpose(2, 0, 1)
  
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

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Load the model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# Only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)


# Train model
nEpochs = 75
batchSize = 4
learningRate = 1e-5
optimizer = Adam(model.mask_decoder.parameters(), lr=learningRate, weight_decay=0.01)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print('Device:',device, ':',torch.cuda.device_count())

#neptune:
run = neptune.init_run(
    project="MRI-infarct-segmentation/FineTuneMedSam-JAS",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkODc2ZGIzZC0wMjllLTQ2OGEtODlmZC05MGNmZDg2YzYwMWUifQ==",
)
params = {"Data": "JAS", "Model": "MedSam","Learning_rate": learningRate, "Optimizer": "Adam", "Lossfunction": "DiceCELoss",}
run["parameters"] = params

best_val_loss = 1
for iEpoch in range(nEpochs):
      print(f'------- Epoch: {iEpoch+1} of {nEpochs} -------')
      print('Training:\n')
      type = "train"
      model.train()
      data = train_data
      Epoch_train_losses = []
      for i in range(0,len(data)):
            print(f'Volume: {i+1} of {len(data)}')

            #Load data:
            dataset = getVolume(i)
            train_dataset = SAMDataset(dataset = dataset, processor = processor, type = type)
            train_dataloader = DataLoader(train_dataset, batch_size = batchSize, shuffle = True, drop_last=False)

            # Train loop:
            volume_losses = []
            for batch in tqdm(train_dataloader):
                # forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=batch["pixel_values"].to(device), input_boxes=batch["input_boxes"].to(device), multimask_output=False)
                torch.cuda.empty_cache()

                # compute loss
                with torch.cuda.amp.autocast():
                    predicted_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                    loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                torch.cuda.empty_cache()

                # backward pass 
                optimizer.zero_grad()
                loss.backward() #(compute gradients of parameters w.r.t. loss)
                optimizer.step() # optimize
                torch.cuda.empty_cache()

                volume_losses.append(loss.item())
            vol_train_loss = mean(volume_losses)
            Epoch_train_losses.append(vol_train_loss)
            print(f'Train loss for volume {i}: {vol_train_loss}\n')
            run["Training/train_loss"].append(vol_train_loss)

      # Validation:
      print('______________________')
      print('Validation:\n')
      type = "validation"
      model.eval()
      data = val_data
      Epoch_val_losses = []
      for i in range(0,len(data)):
            print(f'Volume: {i+1} of {len(data)}')

            #Load data:
            dataset = getVolume(i)
            val_dataset = SAMDataset(dataset = dataset, processor = processor, type = type)
            val_dataloader = DataLoader(val_dataset, batch_size = batchSize, shuffle = False, drop_last=False)

            volume_losses = []
            for batch in tqdm(val_dataloader):
                # forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=batch["pixel_values"].to(device), input_boxes=batch["input_boxes"].to(device), multimask_output=False)
                torch.cuda.empty_cache()

                # compute loss
                with torch.cuda.amp.autocast():
                    predicted_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                    loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                torch.cuda.empty_cache()

                # backward pass 
                optimizer.zero_grad()
                loss.backward() #(compute gradients of parameters w.r.t. loss)
                optimizer.step() # optimize
                torch.cuda.empty_cache()
                
                volume_losses.append(loss.item())
            vol_val_loss = mean(volume_losses)
            Epoch_val_losses.append(vol_val_loss)
            print(f'Validation loss for volume {i}: {vol_val_loss}\n')
            run["Training/val_loss"].append(vol_val_loss)

      #Epoch:
      train_loss = mean(Epoch_train_losses)
      val_loss = mean(Epoch_val_losses)
      if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/home/rosengaard/mri-infarct-segmentation/Anders/V4/weights/model_checkpointV2.pth")
            print('New model saved with best val loss:', best_val_loss)

      print('______________________________________________')
      print('Epoch', iEpoch+1, 'Complete:')
      print(f'Epoch train loss: {train_loss}')
      print(f'Epoch val loss: {val_loss}')
      print('______________________________________________\n')
      run["Training/epoch_train_loss"].append(train_loss)
      run["Training/epoch_val_loss"].append(val_loss)

run.stop()
print('______________________________________________')
print('Training ended')