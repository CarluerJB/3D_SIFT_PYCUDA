import os
import numpy as np
import sys
import nibabel as nib
from numba import *
from numba import cuda
from src_common import *
import nipy as nipy
import pycuda.driver as pycuda
import pycuda.autoinit, pycuda.compiler
from SIFT_Class import FEATUREIO

#Specify wich function will be run on GPU
#mandel_gpu = cuda.jit(device=True)(blur_3d_simpleborders_CUDA)


if(len(sys.argv)<3):
    print_options()
    raise SystemExit

bDoubleImageSize = 0;
bOutputText = 1;
bWorldCoordinates = 0;
bIsotropicProcessing = 0;
iFlipCoord = 0; # Flip a coordinate system: x=1,y=2,z=3
fEigThres = 140;
bMultiModal = 0;
witchCuda = 0;

option_list=[]
for args in sys.argv:
    if args[0]=='-':
        option_list.append(str(args[1:]))
enabled_option=0
for option in option_list:
    if option == "2-":
        bDoubleImageSize = -1
        enabled_option+=1
    if option == "2+" or option == "2":
        bDoubleImageSize = 1
        enabled_option+=1
    if option == "c" or option == "C":
        witchCuda = 1
        enabled_option+=1
    if option == 'w' or option == 'W':
        bWorldCoordinates = 1
        bIsotropicProcessing = 1
        enabled_option+=1
        if option[2] == 's' or option[2] == 'S':
            # Optionally use the nifti sform coordinate system, if available
            bWorldCoordinates = 2
            enabled_option+=1
    if enabled_option==0:
        print( "Error: unknown command line argument: " + str(option));
        print_options();
print("Extracting features: " + str(sys.argv[len(option_list)+1]))

fioIn = nibabel.load(sys.argv[len(option_list)+1])
# Define initial image pixel size
fInitialBlurScale = 1.0

if( bDoubleImageSize != 0 ):
    if( bDoubleImageSize == 1 ):
        # Performing image doubling halves the size of pixels
        fioIn = fioDoubleSize(fioIn)
        show_image(fioIn)
        fInitialBlurScale *= 0.5
    elif( bDoubleImageSize == -1 ):
        # Reduce image size, initial pi
        fioTmp = fioIn
        fioIn.shape[0] /= 2
        fioIn.shape[1] /= 2
        fioIn.shape[2] /= 2
        fioIn=fioSubSample2DCenterPixel( fioTmp, fioIn )
if fioIn.shape[2] <=1:
    print("Could not read volume: " + str(sys.argv[len(option_list)+1]))
print("Input image: i=" + str(fioIn.shape[0]) + " j=" + str(fioIn.shape[1]) + " k=" + str(fioIn.shape[2]) + "\n")
vecFeats3D=[]

#Allocation of inage on GPU
fioInGPU=FEATUREIO()
fioInGPU.x=fioIn.shape[0]
fioInGPU.y=fioIn.shape[1]
fioInGPU.z=fioIn.shape[2]
fioInGPU.iFeaturesPerVector=1
data=fioIn.get_fdata()[:,:,:]
data=data.astype(np.float32)
data=data.ravel(order='F')

fioInGPU.data = gpuarray.to_gpu(data)
fioInGPU._cpu_data=data

vecFeats3D=msGeneratePyramidDOG3D_efficient_test(fioInGPU, vecFeats3D, fInitialBlurScale, 0, fEigThres)
fSizeFactor=1
if bDoubleImageSize>0:
    fSizeFactor /= 2
elif bDoubleImageSize<0:
    fSizeFactor *= 2
for Feat in vecFeats3D:
    Feat.NormalizeData()
    Feat=msResampleFeaturesGradientOrientationHistogram(Feat)
    Feat.NormalizeDataRankedPCs()
    Feat.x *= fSizeFactor
    Feat.y *= fSizeFactor
    Feat.z *= fSizeFactor
    Feat.scale *= fSizeFactor
if bOutputText:
    ppcComments=[]
    ppcComments.append("# Extraction Voxel Resolution (ijk) : " + str(fioIn.shape[0]) + " " + str(fioIn.shape[1])  + " " + str(fioIn.shape[2]) + "\n")
    #ppcComments.append("Extraction Voxel Size (mm)  (ijk) : " + str(fioIn.affine[0,0]) + str(fioIn.affine[1,1]) + str(fioIn.affine[2,2]))
    ppcComments.append("# Feature Coordinate Space: millimeters (qto_xyz) : ")
    for line in fioIn.affine:
        for col in line:
            ppcComments[1]+=str(round(col,6))+" "
    ppcComments[1]+="\n"
    msFeature3DVectorOutputText( vecFeats3D, sys.argv[len(option_list)+2], fEigThres, ppcComments )
""" TO SHOW SLICES
imgdata = fioIn.get_fdata()
slice_0 = imgdata[130, :, :]
slice_1 = imgdata[:, 150, :]
slice_2 = imgdata[:, :, 130]
show_slices([slice_0, slice_1, slice_2])
"""
