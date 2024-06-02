import os
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import scipy
from scipy.ndimage import morphology
from scipy import stats
from scipy.optimize import curve_fit
from scipy.ndimage import morphology, gaussian_filter
from dipy.align import (affine_registration, center_of_mass, translation, rigid, affine)
from dipy.align.imaffine import AffineMap

current_directory = os.environ['HOME']

# Load model
Lesion_model_name = os.path.join(current_directory, "mri-infarct-segmentation/Anders/V1/DAGMNET_v1.3/ADSv1.3/data/Trained_Nets/DAGMNet_CH3.h5")
Lesion_model = load_model(Lesion_model_name, compile=False)
N_channel = 3

#Load data
test_data_path = os.path.join(current_directory, 'mri-infarct-segmentation/data/DUPONT/datasplit/FixedPath/test_data.csv')
df_test = pd.read_csv(test_data_path)
data = []
for i in range(0, len(df_test)):
    data.append([os.path.join(current_directory, df_test["DWI_path"][i]),
                os.path.join(current_directory, df_test["ADC_path"][i]),
                os.path.join(current_directory, df_test["b0"][i]),
                os.path.join(current_directory, df_test["Label_path"][i])])
    
def preprocess_pipeline(ADCPath,
                        DwiPath,
                        B0Path,
                        N_channel,
                        level_iters = [1000,100], # iterations for sequancial affine registration
                        sigmas = [3.0,1.0], # parameters for sequancial affine registration
                        factors = [4,2], # parameters for sequancial affine registration
                        bvalue = 1000,
                        MNI_spec = 'JHU_MNI_181',
                        ):
    
    # loading dwi, b0, ADC
    Dwi_imgJ, Dwi_img, _ = load_img_AffMat(DwiPath)
    B0_imgJ, B0_img, B0_AffMat = load_img_AffMat(B0Path)
    ADC_imgJ, ADC_img, _ = load_img_AffMat(ADCPath)
    
    # loading template
    TemplateDir = "/home/rosengaard/mri-infarct-segmentation/Anders/V1/DAGMNET_v1.3/ADSv1.3/data/template"
    JHU_B0_withskull_fnamePth = os.path.join(TemplateDir, 'JHU_SS_b0_padding.nii.gz')
    JHU_B0_imgJ, JHU_B0_img, JHU_B0_AffMat = load_img_AffMat(JHU_B0_withskull_fnamePth)
     
    pixdim = Dwi_imgJ.header.get_zooms()
    VolSpacing = 1
    for _ in range(3):
        i = pixdim[_]
        if i !=0:
            VolSpacing = VolSpacing*i
 
    # Mapping to MNI with skull for MaskNet:
    level_iters = [1] # 15'
    sigmas = [6.0]
    factors = [2] # 2
    B0_MNI_img, reg_affine, affine_map = Sequential_Registration_b0(static=JHU_B0_img, 
                                                                static_grid2world=JHU_B0_AffMat,
                                                                moving=B0_img,
                                                                moving_grid2world=B0_AffMat,
                                                                level_iters = level_iters, 
                                                                sigmas = sigmas, 
                                                                factors = factors
                                                               )
    
    Dwi_MNI_img = affine_map.transform(Dwi_img)
    
    # Loading MaskNet:
    BrainMaskNet_path = "/home/rosengaard/mri-infarct-segmentation/Anders/V1/DAGMNET_v1.3/ADSv1.3/data/Trained_Nets/BrainMaskNet.h5"
    MaskNet = load_model(BrainMaskNet_path, compile=False)
    
    # Get brain mask in raw space:
    mask_MNI_img = get_MaskNet_MNI(MaskNet, Dwi_MNI_img, B0_MNI_img)
    mask_raw_img = affine_map.transform_inverse((mask_MNI_img>0.5)*1, interpolation='nearest')
    mask_raw_img = (mask_raw_img>0.5)*1.0
    
    # Get skull stripped DWI, ADC and b0:
    Dwi_ss_img = Dwi_img*mask_raw_img
    B0_ss_img = B0_img*mask_raw_img
    ADC_ss_img = ADC_img*mask_raw_img
    
    # Loading template_ss 
    JHU_B0_ss_fnamePth = os.path.join(TemplateDir, 'JHU_SS_b0_ss_padding.nii.gz')  
    JHU_B0_ss_imgJ, JHU_B0_ss_img, JHU_B0_ss_AffMat = load_img_AffMat(JHU_B0_ss_fnamePth)
    
    # Get mapping to MNI without skull for lesion detection model
    B0_ss_MNI_img, reg_affine, affine_map = Sequential_Registration_b0(static=JHU_B0_ss_img, 
                                                                    static_grid2world=JHU_B0_ss_AffMat,
                                                                    moving=B0_ss_img,
                                                                    moving_grid2world=B0_AffMat,
                                                                    level_iters = level_iters, 
                                                                    sigmas = sigmas, 
                                                                    factors = factors
                                                                   )
    
    # Mapping to MNI without skull for lesion detection model 
    Dwi_ss_MNI_img = affine_map.transform(Dwi_ss_img)
    ADC_ss_MNI_img = affine_map.transform(ADC_ss_img)
    mask_raw_MNI_img = affine_map.transform(mask_raw_img, interpolation='nearest') 
    mask_raw_MNI_img = (mask_raw_MNI_img>0.5)*1.0
    
    # Normalizing DWI
    Dwi_ss_MNI_norm_img = get_dwi_normalized(Dwi_ss_MNI_img,mask_raw_MNI_img)
    
    # get Prob. IS
    if N_channel==3:
        #print('------ Calculating Prob. IS Map for CH3 ------')
        Prob_IS = get_Prob_IS(Dwi_ss_MNI_norm_img, ADC_ss_MNI_img,  mask_raw_MNI_img, TemplateDir)
    else:
        #print('------ N_channel==2 ------')
        Prob_IS = None
        
    # Get standard normalization within brainmask:
    tmp = Dwi_ss_MNI_norm_img[mask_raw_MNI_img>0.5]
    Dwi_ss_MNI_BSN_img = (Dwi_ss_MNI_norm_img - np.mean(tmp)) / np.std(tmp)

    tmp = ADC_ss_MNI_img[mask_raw_MNI_img>0.5]
    ADC_ss_MNI_BSN_img = (ADC_ss_MNI_img - np.mean(tmp)) / np.std(tmp)

    return Dwi_ss_MNI_BSN_img, ADC_ss_MNI_BSN_img, Prob_IS, affine_map,mask_raw_img, mask_raw_MNI_img

def postprocess(modeloutput, affine_map, mask_raw_img, mask_raw_MNI_img):
    stroke_pred_img = modeloutput
    stroke_pred_img = stroke_pred_img*mask_raw_MNI_img
    
    # Map lesion back to raw space:
    stroke_pred_raw_img = affine_map.transform_inverse((stroke_pred_img>0.5)*1, interpolation='nearest')
    stroke_pred_raw_img = stroke_pred_raw_img*mask_raw_img
    stroke_pred_raw_img = (stroke_pred_raw_img>0.5)*1
    
    # Morpology operations:
    stroke_pred_raw_img = remove_small_objects_InSlice(stroke_pred_raw_img)
    stroke_pred_raw_img = Stroke_closing(stroke_pred_raw_img)
    stroke_pred_raw_img = morphology.binary_fill_holes(stroke_pred_raw_img)

    return stroke_pred_raw_img

def load_img_AffMat(img_fnamePth):
    imgJ = nib.load(img_fnamePth)
    img = np.squeeze(imgJ.get_fdata())*1.0 # change data format to floating point
    img_AffMat = imgJ.affine
    return imgJ, img, img_AffMat

def Sequential_Registration_b0(static, 
                               static_grid2world, 
                               moving, 
                               moving_grid2world, 
                               level_iters = [100,100], # iterations for sequancial affine registration
                               sigmas = [3.0,1.0], # parameters for sequancial affine registration
                               factors = [4,2], # parameters for sequancial affine registration
                              ):
    # sequantial affine registration b0 images
    pipeline = [center_of_mass, translation, rigid, affine]
    xformed_img, reg_affine = affine_registration(
        moving,
        static,
        moving_affine=moving_grid2world,
        static_affine=static_grid2world,
        nbins=16,
        metric='MI',
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors)
    affine_map = AffineMap(reg_affine,
                       static.shape, static_grid2world,
                       moving.shape, moving_grid2world)
    return xformed_img, reg_affine, affine_map

def Stroke_closing(img):
    # used to close stroke prediction image
    new_img = np.zeros_like(img)
    new_img = morphology.binary_closing(img, structure=np.ones((2,2,2)))
    return new_img

def get_MaskNet_MNI(model, Dwi_MNI_img, B0_MNI_img):
    # To inference brain mask from MaskNet model
    # model specifies which pre-trained DL model is used to inference
    # Dwi_MNI_img and B0_MNI_img are input images in MNI domain
    
    # Down sampling
    dwi = Dwi_MNI_img[0::4,0::4,0::4,np.newaxis] # Down sample for MaskNet, dim should be [48, 56, 48, 1]
    dwi  = (dwi-np.mean(dwi))/np.std(dwi)

    b0 = B0_MNI_img[0::4,0::4,0::4, np.newaxis] # Down sample for MaskNet, dim should be [48, 56, 48, 1]
    b0  = (b0-np.mean(b0))/np.std(b0)
    x = np.expand_dims(np.concatenate((dwi,b0),axis=3), axis=0)

    # inference
    y_pred = model.predict(x, verbose=0)
    y_pred = (np.squeeze(y_pred)>0.5)*1.0

    
    # the following is post processing of predicted mask by 
    # 1) selecting the major non-zero voxel
    # 2) closing
    # 3) binary fill holes
    # 4) upsampling to high resolution space by (4,4,4)
    
    mask_label, num_features = scipy.ndimage.measurements.label(y_pred)
    dilate_mask = (mask_label == scipy.stats.mode(mask_label[mask_label>0].flatten())[0][0])*1
    dilate_mask = Stroke_closing(dilate_mask)
    dilate_mask = morphology.binary_fill_holes(dilate_mask)
    upsampling_mask = np.repeat(np.repeat(np.repeat(dilate_mask, 4, axis=0), 4, axis=1), 4, axis=2)

    return upsampling_mask

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def get_dwi_normalized(Dwi_ss_MNI_img,mask_raw_MNI_img):
    # bimodal gaussion fitting normalization
    Dwi_d = Dwi_ss_MNI_img[mask_raw_MNI_img>0.5]

    md = scipy.stats.mode(Dwi_d.astype('int16'))[0][0]
    if md > np.mean(Dwi_d):
        p0_mu = md
    else:
        p0_mu = np.mean(Dwi_d)
        
    Dwi_hist, xData = np.histogram(Dwi_d, bins=np.arange(np.max(Dwi_d)),  density=True)
    xData=(xData[1:]+xData[:-1])/2 # for len(x)==len(y)
    try : 
        bounds = ([0, 0, -np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, p0_mu, np.inf, np.inf])
        params,cov=curve_fit(bimodal,xData,Dwi_hist, bounds=bounds, p0=(p0_mu,1,1, 0,1,1 ), maxfev=5000)
        mu1 = params[0]
        sigma1 = params[1]
        a1 = params[2]
    except : 
        mu1 = p0_mu
        sigma1 = np.std(Dwi_d)
    
    Dwi_ss_MNI_norm_img = (Dwi_ss_MNI_img-mu1)/sigma1
    return Dwi_ss_MNI_norm_img

def qfunc(x):
    return 0.5-0.5*scipy.special.erf(x/np.sqrt(2))

def get_Prob_IS(Dwi_ss_MNI_norm_img, ADC_ss_MNI_img, mask_raw_MNI_img, 
                TemplateDir, 
                model_vars = [2,1.5,4,0.5,2,2]):
    # get initail guess image for ischemic stroke
    template_fnamePth = os.path.join(TemplateDir, 'normal_mu_dwi_Res_ss_MNI_scaled_normalized.nii.gz')   
    _, normal_dwi_mu_img, _ = load_img_AffMat(template_fnamePth)

    template_fnamePth = os.path.join(TemplateDir, 'normal_std_dwi_Res_ss_MNI_scaled_normalized.nii.gz')   
    _, normal_dwistd_img, _ = load_img_AffMat(template_fnamePth)

    template_fnamePth = os.path.join(TemplateDir, 'normal_mu_ADC_Res_ss_MNI_normalized.nii.gz')   
    _, normal_adc_mu_img, _ = load_img_AffMat(template_fnamePth)

    template_fnamePth = os.path.join(TemplateDir, 'normal_std_ADC_Res_ss_MNI_normalized.nii.gz')   
    _, normal_adc_std_img, _ = load_img_AffMat(template_fnamePth)
    
    fwhm = model_vars[0]
    g_sigma = fwhm/2/np.sqrt(2*np.log(2))
    alpha_dwi = model_vars[1]
    lambda_dwi = model_vars[2]
    alpha_adc = model_vars[3]
    lambda_adc = model_vars[4]
    id_isch_zth = model_vars[5]

    img = (Dwi_ss_MNI_norm_img - np.mean(Dwi_ss_MNI_norm_img)) / np.std(Dwi_ss_MNI_norm_img)
    for i in range(img.shape[-1]):
        img[:,:,i] = gaussian_filter(img[:,:,i], g_sigma)
    dissimilarity = np.tanh((img - normal_dwi_mu_img)/normal_dwistd_img/alpha_dwi)
    dissimilarity[dissimilarity<0] = 0
    dissimilarity = dissimilarity ** lambda_dwi
    dissimilarity[Dwi_ss_MNI_norm_img<id_isch_zth] = 0
    dwi_H2 = dissimilarity*(mask_raw_MNI_img>0.49)*1.0

    img = (ADC_ss_MNI_img - np.mean(ADC_ss_MNI_img)) / np.std(ADC_ss_MNI_img)
    for i in range(img.shape[-1]):
        img[:,:,i] = gaussian_filter(img[:,:,i], g_sigma)
    dissimilarity = np.tanh((img - normal_adc_mu_img)/normal_adc_std_img/alpha_adc)
    dissimilarity[dissimilarity>0] = 0
    dissimilarity = (-dissimilarity) ** lambda_adc
    adc_H1 = dissimilarity*(mask_raw_MNI_img>0.49)*1.0


    id_isch = Dwi_ss_MNI_norm_img
    id_isch = (1-qfunc(id_isch/id_isch_zth))*(id_isch>id_isch_zth)

    Prob_IS = dwi_H2*adc_H1*id_isch*(mask_raw_MNI_img>0.49)*1.0
    
    return Prob_IS

def remove_small_objects(img, remove_max_size=5, structure = np.ones((3,3))):
    # remove small objects in prediction
    binary = img
    binary[binary>0] = 1
    labels = np.array(scipy.ndimage.label(binary, structure=structure))[0]
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
    new_img = img
    for index in np.unique(labels):
        if labels_num[index]<remove_max_size:
            new_img[labels==index] = 0
    return new_img

def remove_small_objects_InSlice(img, remove_max_size=5,  structure = np.ones((3,3))):
    # remove small objects in prediction for each slice
    img = np.squeeze(img)
    new_img = np.zeros_like(img)
    
    for idx in range(img.shape[-1]):
        #new_img[:,:,idx] = morphology.remove_small_objects(img[:,:,idx],remove_max_size=remove_max_size, structure=structure)
        new_img[:,:,idx] = remove_small_objects(img[:,:,idx],remove_max_size=remove_max_size, structure=structure)
    return new_img

def get_stroke_seg_MNI(model, dwi_img, adc_img, Prob_IS=None, N_channel=3, DS=2):
    # To repeatedly inference and stack the stroke lesion mask
    stroke_pred_resampled =  np.squeeze(np.zeros_like(dwi_img))
    for x_idx, y_idx, slice_idx in [(x,y,z) for x in range(DS) for y in range(DS) for z in range(2*DS)]:
        if N_channel==3:
            dwi_DS_img = dwi_img[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            adc_DS_img = adc_img[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            Prob_IS_DS_img = Prob_IS[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            imgs_input = np.expand_dims(np.concatenate((dwi_DS_img,adc_DS_img,Prob_IS_DS_img),axis=3), axis=0)
        elif N_channel==2:
            dwi_DS_img = dwi_img[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            adc_DS_img = adc_img[x_idx::DS,y_idx::DS,slice_idx::2*DS, np.newaxis]
            imgs_input = np.expand_dims(np.concatenate((dwi_DS_img,adc_DS_img),axis=3), axis=0)
        stroke_pred = model.predict(imgs_input, verbose=0)[0]
        stroke_pred = np.squeeze(stroke_pred)
        stroke_pred_resampled[x_idx::DS,y_idx::DS,slice_idx::2*DS] = stroke_pred

    # the following is post processing predicted mask by 
    # 1) closing
    # 2) binary fill holes
    stroke_pred_tmp = (stroke_pred_resampled>0.5)
    stroke_pred_tmp = Stroke_closing(stroke_pred_tmp)
    stroke_pred_tmp = morphology.binary_fill_holes(stroke_pred_tmp)
    
    return stroke_pred_tmp

def MNIdePadding_imgJ(imgJ):
    img = np.squeeze(imgJ.get_fdata())
    de_img = img[5:186, 3:220, 5:186]
    
    deimg_AffMatt = imgJ.affine
    deimg_AffMatt[0,3] = 90
    deimg_AffMatt[1,3] = -108
    deimg_AffMatt[2,3] = -90
    
    img_header = imgJ.header
    img_header['glmax'] = np.max(de_img)
    img_header['glmin'] = np.min(de_img)
    img_header['xyzt_units']=0
    img_header['descrip']='ADS May 03 2022'
    de_imgJ = nib.Nifti1Image(de_img, deimg_AffMatt, img_header)
    de_imgJ.header.set_slope_inter(1, 0)
    return de_imgJ

def set_IMG_toMNI_spec(imgJ, MNI_spec='JHU_MNI_181'):
    img = np.squeeze(imgJ.get_fdata())
    if MNI_spec == 'JHU_MNI_181':
        img_ = img
        img_AffMatt = imgJ.affine
        img_AffMatt[0,3] = 0
        img_AffMatt[1,3] = 0
        img_AffMatt[2,3] = 0
    elif MNI_spec == 'JHU_MNI_182':
        img_ = np.zeros((182, 218, 182)).astype(img.dtype)
        img_[0:181, 0:217, 0:181] = img
        img_AffMatt = imgJ.affine
        img_AffMatt[0,3] = 0
        img_AffMatt[1,3] = 0
        img_AffMatt[2,3] = 0
    elif MNI_spec == 'MNI152_181':
        img_ = img
        img_AffMatt = imgJ.affine
        img_AffMatt[0,3] = 90
        img_AffMatt[1,3] = -126
        img_AffMatt[2,3] = -72
    elif MNI_spec == 'MNI152_182':
        img_ = np.zeros((182, 218, 182)).astype(img.dtype)
        img_[0:181, 0:217, 0:181] = img
        img_AffMatt = imgJ.affine
        img_AffMatt[0,3] = 90
        img_AffMatt[1,3] = -126
        img_AffMatt[2,3] = -72
    else:
        print("Wrong MNI_spec setting, defaut setting (JHU_MNI_181) will be used.")
        img_ = img
        img_AffMatt = imgJ.affine
        img_AffMatt[0,3] = 0
        img_AffMatt[1,3] = 0
        img_AffMatt[2,3] = 0
        
    img_header = imgJ.header
    img_header['glmax'] = np.max(img_)
    img_header['glmin'] = np.min(img_)
    img_header['xyzt_units']=0
    img_header['descrip']='ADS May 03 2022'
    imgJ_ = nib.Nifti1Image(img_, img_AffMatt, img_header)
    imgJ_.header.set_slope_inter(1, 0)
    return imgJ_

def get_new_NibImgJ(new_img, temp_imgJ, dataType=np.float32):
    temp_imgJ.set_data_dtype(dataType)
    if new_img.dtype !=  np.dtype(dataType):
        new_img.astype(np.dtype(dataType))
    img_header = temp_imgJ.header
    img_header['glmax'] = np.max(new_img)
    img_header['glmin'] = np.min(new_img)
    img_header['xyzt_units']=0
    img_header['descrip']='ADS May 03 2022'
    new_imgJ = nib.Nifti1Image(new_img,temp_imgJ.affine,img_header)
    new_imgJ.header.set_slope_inter(1, 0)
    return new_imgJ

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

def Find_DICE_vol(segmentation_mask, label_mask):
    segmentation_mask = segmentation_mask > 0.5 # binary
    areaOfOverlap = np.sum(segmentation_mask * label_mask).item()
    totalArea = np.sum(segmentation_mask).item() + np.sum(label_mask).item()
    dice_score = (2*areaOfOverlap)/totalArea
    return dice_score

def plotFunc(DWI_slice, modelout, Label_slice):
    fig, axes = plt.subplots(1, 3, figsize=(7, 7))  # Adjust figsize as needed
    axes[0].imshow(DWI_slice, cmap='gray')  # Assuming grayscale images
    axes[0].set_title('DWI slice')
    axes[0].axes.axis('off')
    axes[1].imshow(modelout, cmap='gray')  # Assuming grayscale images
    axes[1].set_title('DAGMNET output')
    axes[1].axes.axis('off')
    axes[2].imshow(Label_slice, cmap='gray')  # Assuming grayscale images
    axes[2].set_title('Label')
    axes[2].axes.axis('off')
    plt.show()