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
#import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.imagenet_utils import preprocess_input
from transformers import SamModel, SamConfig, SamProcessor
from segment_anything import sam_model_registry
from segment_anything.modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer)
import monai
from datasets import Dataset as _Dataset
import neptune
from tqdm import tqdm

affreg = AffineRegistration(level_iters=[0])
current_directory = os.environ['HOME']

#Load #MedSAM model:
MedSAM_CKPT_PATH = "/home/rosengaard/mri-infarct-segmentation/Anders/V2/Models/MedSamSymDif/MedSamWeights/medsam_vit_b.pth" #"medsam_vit_b.pth" #Pretrained model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #device = "cuda:0"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

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
def loadVolume(vol_path, expand):
    vol = nib.load(vol_path).get_fdata()
    vol = ((vol - np.min(vol)) / (np.max(vol) - np.min(vol))).astype(float)
    if expand:
        vol = np.expand_dims(vol, axis=2)
    return vol

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

def create_DifDif_vol(DWI_vol, ADC_vol, b0_vol):
    # Compute the symmetric difference of left and right hemisphere from DWI:
    symDif_vol = Create_DWI_symDif_map(DWI_vol)

    # Th:
    symDif_vol = np.where(((symDif_vol*-1+1) < DWI_vol), DWI_vol, DWI_vol*0.5)

    # Compute the difference between the modalities ADC and b0:
    b0_th = np.where(DWI_vol > (np.mean(DWI_vol)), b0_vol, 0)
    ADC_th = np.where(DWI_vol > (np.mean(DWI_vol)), ADC_vol, 0)
    modDif_vol = np.where(b0_th > ADC_th , DWI_vol, DWI_vol * 0.5)

    # Compute a dif_dif slice based on the difference between the to dif_slices:
    DifDif_vol = np.asarray(symDif_vol * modDif_vol)
    
    # Norm:
    DifDif_vol = ((DifDif_vol - DifDif_vol.min()) / (DifDif_vol.max() - DifDif_vol.min()) * 255).astype(float)

    return DifDif_vol

def get_volume(DWI_path, ADC_path, b0_path, channels):
    # Stack slices into 3-channel slice:
    if channels == "1C=DWI":
        DWI_vol = loadVolume(DWI_path, True)
        Stacked_vol = np.repeat(DWI_vol, 3, axis=2)
    if channels == "3C=DWI_ADC_b0":
        DWI_vol = loadVolume(DWI_path, True)
        ADC_vol = loadVolume(ADC_path, True)
        b0_vol = loadVolume(b0_path, True)
        Stacked_vol = np.concatenate((DWI_vol, ADC_vol), axis=2)
        Stacked_vol = np.concatenate((Stacked_vol, b0_vol), axis=2)
    if channels == "3C=DWI_ADC_DifDif":
        DWI_vol = loadVolume(DWI_path, False)
        ADC_vol = loadVolume(ADC_path, False)
        b0_vol = loadVolume(b0_path, False)
        DifDif_vol = create_DifDif_vol(DWI_vol, ADC_vol, b0_vol)

        DWI_vol = np.expand_dims(DWI_vol, axis=2)
        ADC_vol = np.expand_dims(ADC_vol, axis=2)
        DifDif_vol = np.expand_dims(DifDif_vol, axis=2)
        
        Stacked_vol = np.concatenate((DWI_vol, ADC_vol), axis=2)
        Stacked_vol = np.concatenate((Stacked_vol, DifDif_vol), axis=2)

    return Stacked_vol

def plotFunc(DWI_slice, model_output, label_slice, bbox):
    fig, axes = plt.subplots(1, 3, figsize=(9, 9))
    axes[0].imshow(DWI_slice, cmap='gray')
    axes[0].set_title('DWI slice')
    axes[0].axis('off')
    axes[0].plot([bbox[0], bbox[2]],[bbox[1], bbox[1]], color='green', linewidth=1)
    axes[0].plot([bbox[0], bbox[2]],[bbox[3], bbox[3]], color='green', linewidth=1)
    axes[0].plot([bbox[0], bbox[0]],[bbox[1], bbox[3]], color='green', linewidth=1)
    axes[0].plot([bbox[2], bbox[2]],[bbox[1], bbox[3]], color='green', linewidth=1)

    axes[1].imshow(model_output, cmap='gray')
    axes[1].set_title('MedSAM output')
    axes[1].axis('off')
    axes[2].imshow(label_slice, cmap='gray')
    axes[2].set_title('Label')
    axes[2].axis('off')
    plt.show()

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
            perturbation = 0
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

def Find_DICE_slice(segmentation_mask, label_mask):
    if np.sum(label_mask).item() > 0: #Because this works on slice basics and not all slices has an infact...
        segmentation_mask = segmentation_mask > 0.5 # binary
        areaOfOverlap = np.sum(segmentation_mask * label_mask).item()
        totalArea = np.sum(segmentation_mask).item() + np.sum(label_mask).item()
        #totalArea = np.sum(combinedSegmask[:,:,0]).item() + np.sum(label_slice).item()
        dice_score = (2*areaOfOverlap)/totalArea
    else:
        dice_score = 0 # Dummy

    return dice_score

class MSSD:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.image = None
        self.image_embeddings = None
        self.img_size = None
        self.x_min, self.x_max, self.y_min, self.y_max = 0., 0., 0., 0.

    @torch.no_grad()
    def _infer(self, bbox):
        ori_H, ori_W = self.img_size
        scale_to_1024 = 1024 / np.array([ori_W, ori_H, ori_W, ori_H])
        bbox_1024 = bbox * scale_to_1024
        bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float).unsqueeze(0).to(self.model.device)
        if len(bbox_torch.shape) == 2:
            bbox_torch = bbox_torch.unsqueeze(1)
    
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None,boxes=bbox_torch,masks=None,)
        low_res_logits, _ = self.model.mask_decoder(image_embeddings = self.image_embeddings, # (B, 256, 64, 64)
                                                    image_pe = self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                                                    sparse_prompt_embeddings = sparse_embeddings, # (B, 2, 256)
                                                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                                                    multimask_output=False,
                                                    )
        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(low_res_pred, size=self.img_size, mode="bilinear", align_corners=False)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

    def procesInput(self, image):
        self.image = image
        self.img_size = image.shape[:2]

        img_resize = cv2.resize(image,(1024, 1024),interpolation=cv2.INTER_CUBIC)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        # convert the shape to (3, H, W)
        assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)
        image_preprocess = img_tensor
        
        with torch.no_grad():
            self.image_embeddings = self.model.image_encoder(image_preprocess)
        
        bbox = np.array([self.x_min, self.y_min, self.x_max, self.y_max])
        with torch.no_grad():
            seg = self._infer(bbox)
            torch.cuda.empty_cache()

        color = np.array([251/255, 252/255, 30/255, 0.95])
        h, w = seg.shape[-2:]
        mask_image = seg.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image

    def setbbox_fromPromt(self, _bbox):
        self.x_min = _bbox[0]
        self.y_min = _bbox[1]
        self.x_max = _bbox[2]
        self.y_max = _bbox[3]

    def get_segmentationsMask(self, image, bbox):
        results = []
        self.setbbox_fromPromt(bbox)
        results.append(self.procesInput(image))
        return results