import nibabel as nib
import nibabel.processing
import math
from itertools import *
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
from numba import vectorize
import nilearn
from nilearn.image import resample_img
from scipy.signal import resample_poly
from scipy.ndimage import zoom
from SIFT_Class import *
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from scipy import misc
import SIFT_Class as sift
import os
import time
import gputools
import scipy.stats as ss
from scipy.signal import argrelextrema
import pycuda.driver as pycuda
import pycuda.autoinit, pycuda.compiler
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from package import gpuFunc as cuda


BLURS_PER_OCTAVE = 3
BLURS_EXTRA = 3
BLURS_TOTAL = BLURS_PER_OCTAVE+BLURS_EXTRA
BLUR_PRECISION = 0.01
fBlurGradOriHist = 0.5

fHist2ndPeakThreshold = 0.5

def load_img(image):
    img = nib.load(image)
    data = img.get_fdata()
    return img

def print_options_FMM():
	print( "Volumetric Feature matching v1.1\n")
	print( "Determines robust alignment solution mapping coordinates in image 2, 3, ... to image 1.\n")
	print( "Usage: %s [options] <input keys 1> <input keys 2> ... \n", "featMatchMultiple" )
	print( "  <input keys 1, ...>: input key files, produced from featExtract.\n" )
	print( "  <output transform>: output text file with linear transform from keys 2 -> keys 1.\n" )

def print_options():
    print("Volumetric local feature extraction v1.1\n")
    print("Usage: featExtract [options] <input image> <output features>\n")
    print("  <input image>: nifti (.nii,.hdr,.nii.gz).\n")
    print("  <output features>: output file with features.\n")
    print(" [options]\n" )
    print("  -w         : output feature geometry in world coordinates, NIFTI qto_xyz matrix (default is voxel units).\n")
    print("  -2+        : double input image size.\n")
    print("  -2-        : halve input image size.\n")

def get_Param_In_Command(nb_file):
    """
    Paramètre(s) entrée en ligne de commande inséré(s) dans une liste
    INPUT = nombre de fichier à traiter <INT>
    OUTPUT = Les noms de fichiers en .fasta <LIST>
    """
    file = []
    option = []
    for i in range(1, nb_file):
        if "-" not in sys.argv[i]:
            file.append(sys.argv[i])
        else:
            option.append(sys.argv[i])
    return file, option


# detectExtrema4D functions
#
# Detect peak or valley in 4d
#
#

def detectExtrema4D_test(pfioH, pfioC, pfioL, lvaMinima, lvaMaxima, margin=2, bDense=0):

    es = Extremum(pfioH, pfioL, bDense)
    #lvaMinima = regFindFEATUREIOValleys(pfioC.get_fdata()[:,:,:], es, lvaMinima)
    #lvaMinima = ValleyFinderBrutForce(pfioC.get_fdata()[:,:,:], es, lvaMinima)
    #lvaMinima = OLDregFindFEATUREIOValleys(pfioC, es, lvaMinima)
    #lvaMinima = findValleyGPU(pfioC.get_fdata()[:,:,:], es, lvaMinima)
    #lvaMinima = detect_local_minima(pfioC.get_fdata()[:,:,:], es, lvaMinima)
    #lvaMaxima = regFindFEATUREIOPeak(pfioC.get_fdata()[:,:,:], es, lvaMaxima)
    #lvaMaxima = PeakFinderBrutForce(pfioC.get_fdata()[:,:,:], es, lvaMaxima)
    #lvaMaxima = OLDregFindFEATUREIOPeaks(pfioC, es, lvaMaxima)
    #lvaMinima = detect_local_maxima(pfioC.get_fdata()[:,:,:], es, lvaMaxima)

    # Good way to do it but to loud for memory...
    #lvaMaxima, lvaMinima = detect_local_extremum(pfioC.get_fdata()[:,:,:], es, lvaMaxima, lvaMinima)

    # Anotherway GPU oriented
    if pfioL is None:
        lvaMaxima, lvaMinima = detect4D_GPU(pfioC, es, lvaMaxima, lvaMinima, margin)
    else:
        lvaMaxima, lvaMinima = detect4D_GPUFULL(pfioC, es, lvaMaxima, lvaMinima, margin)

    return lvaMinima,lvaMaxima

# using GPU

def detect4D_GPUFULL(fioC, es, lvaMaxima, lvaMinima, margin):
    extremumFunc=cuda.GpuFunc.get_function("d_detectExtrema4D_testFULL")
    fioOut=FEATUREIO()
    fioOut.x=fioC.x
    fioOut.y=fioC.y
    fioOut.z=fioC.z
    fioOut.data= gpuarray.zeros(fioOut.x*fioOut.y*fioOut.z, dtype=np.float32)
    tile_size=10
    cache_size=tile_size + (1 * 2)
    dimBlock=(int(tile_size), int(tile_size), int(tile_size))
    dimCache = int(cache_size*cache_size*cache_size*4)*3
    dimGrid=((math.ceil(fioC.x/float(tile_size))), (math.ceil(fioC.y/float(tile_size))), (math.ceil(fioC.z/float(tile_size))))

    extremumFunc(fioC.data, es.pfioH.data, es.pfioL.data, fioOut.data, np.int32(fioC.x), np.int32(fioC.y), np.int32(fioC.z), np.int32(tile_size), np.int32(cache_size), block=(tile_size,tile_size,tile_size), grid=dimGrid, shared=dimCache)
    extremum_point=np.array(fioOut.x*fioOut.y*fioOut.z, dtype=np.ndarray)
    extremum_point=fioOut.data.get()
    # Naïve method, we get 3d from 1d and then search extremum (slower than the other method)
    """extremum_point = extremum_point.reshape((fioOut.x,fioOut.y,fioOut.z)).transpose((0, 1, 2))

    zMin, yMin, xMin = np.where(extremum_point>(52-margin))
    zMax, yMax, xMax = np.where(extremum_point<(-52+margin))"""
    fioOut._cpu_data=(fioOut.data.get()).astype(np.float32)
    #transform_showCPU(fioOut)
    #SECOND METHOD : use unravel to get 3d index from 1d index => don't need cpu data + reshape + transpose => faster than other tested method
    zMin, yMin, xMin = np.unravel_index(np.where(extremum_point>=(80-margin)), (fioC.x,fioC.y,fioC.z))
    zMax, yMax, xMax = np.unravel_index(np.where(extremum_point<=(-80+margin)), (fioC.x,fioC.y,fioC.z))
    xMin = xMin[0]
    yMin = yMin[0]
    zMin = zMin[0]
    xMax = xMax[0]
    yMax = yMax[0]
    zMax = zMax[0]
    fioC._cpu_data=fioC._cpu_data
    es.pfioH._cpu_data=es.pfioH._cpu_data
    es.pfioH._cpu_data=es.pfioH._cpu_data
    for i in range(len(xMin)):
        lvaMinima.append(Location_Value_XYZ(xMin[i], yMin[i], zMin[i], fioC._cpu_data[xMin[i] + yMin[i]*fioC.x + zMin[i]*fioC.x*fioC.y], es.pfioH._cpu_data[xMin[i] + yMin[i]*fioC.x + zMin[i]*fioC.x*fioC.y], es.pfioH._cpu_data[xMin[i] + yMin[i]*fioC.x + zMin[i]*fioC.x*fioC.y]))
        #print(xMin[i], yMin[i], zMin[i], fioC._cpu_data[xMin[i] + yMin[i]*fioC.x + zMin[i]*fioC.x*fioC.y])
    for i in range(len(xMax)):
        lvaMaxima.append(Location_Value_XYZ(xMax[i], yMax[i], zMax[i], fioC._cpu_data[xMax[i] + yMax[i]*fioC.x + zMax[i]*fioC.x*fioC.y], es.pfioH._cpu_data[xMax[i] + yMax[i]*fioC.x + zMax[i]*fioC.x*fioC.y], es.pfioL._cpu_data[xMax[i] + yMax[i]*fioC.x + zMax[i]*fioC.x*fioC.y]))
        #print(xMax[i], yMax[i], zMax[i], fioC._cpu_data[xMax[i] + yMax[i]*fioC.x + zMax[i]]*fioC.x*fioC.y)
    print("detection of : ", len(xMin), "   Validation of ==>  ", end='')
    print("detection of : ", len(xMax), "   Validation of ==>  ", end='')
    transform_showCPU(fioOut)
    #transform_show(fioOut)
    del fioOut.data
    return lvaMaxima, lvaMinima

def detect4D_GPU(fioC, es, lvaMaxima, lvaMinima, margin=0):
    extremumFunc=cuda.GpuFunc.get_function("d_detectExtrema4D_test")
    fioOut=FEATUREIO()
    fioOut.x=fioC.x
    fioOut.y=fioC.y
    fioOut.z=fioC.z
    fioOut.data= gpuarray.zeros(fioOut.x*fioOut.y*fioOut.z, dtype=np.float32)
    tile_size=10
    cache_size=tile_size + (1 * 2)
    dimBlock=(int(tile_size), int(tile_size), int(tile_size))
    dimCache = int(cache_size*cache_size*cache_size*4)*2
    dimGrid=((math.ceil(fioC.x/float(tile_size))), (math.ceil(fioC.y/float(tile_size))), (math.ceil(fioC.z/float(tile_size))))

    extremumFunc(fioC.data, es.pfioH.data, fioOut.data, np.int32(fioC.x), np.int32(fioC.y), np.int32(fioC.z), np.int32(tile_size), np.int32(cache_size), block=(tile_size,tile_size,tile_size), grid=dimGrid, shared=dimCache)
    extremum_point=np.array(fioOut.x*fioOut.y*fioOut.z, dtype=np.ndarray)
    extremum_point=fioOut.data.get()
    # Naïve method, we get 3d from 1d and then search extremum (slower than the other method)
    """extremum_point = extremum_point.reshape((fioOut.x,fioOut.y,fioOut.z)).transpose((0, 1, 2))

    zMin, yMin, xMin = np.where(extremum_point>(52-margin))
    zMax, yMax, xMax = np.where(extremum_point<(-52+margin))"""
    fioOut._cpu_data=(fioOut.data.get()).astype(np.float32)
    #transform_showCPU(fioOut)
    #SECOND METHOD : use unravel to get 3d index from 1d index => don't need cpu data + reshape + transpose => faster than other tested method
    zMin, yMin, xMin = np.unravel_index(np.where(extremum_point==(53-margin)), (fioC.x,fioC.y,fioC.z))
    zMax, yMax, xMax = np.unravel_index(np.where(extremum_point==(-53+margin)), (fioC.x,fioC.y,fioC.z))
    xMin = xMin[0]
    yMin = yMin[0]
    zMin = zMin[0]
    xMax = xMax[0]
    yMax = yMax[0]
    zMax = zMax[0]

    for i in range(len(xMin)):
        lvaMinima.append(Location_Value_XYZ(xMin[i], yMin[i], zMin[i], fioC._cpu_data[xMin[i] + yMin[i]*fioC.x + zMin[i]*fioC.x*fioC.y]))
        #print(xMin[i], yMin[i], zMin[i], fioC._cpu_data[xMin[i] + yMin[i]*fioC.x + zMin[i]*fioC.x*fioC.y])
    for i in range(len(xMax)):
        lvaMaxima.append(Location_Value_XYZ(xMax[i], yMax[i], zMax[i], fioC._cpu_data[xMax[i] + yMax[i]*fioC.x + zMax[i]*fioC.x*fioC.y]))
        #print(xMax[i], yMax[i], zMax[i], fioC._cpu_data[xMax[i] + yMax[i]*fioC.x + zMax[i]]*fioC.x*fioC.y)
    print("detection of : ", len(xMin), "   Validation of ==>  ", end='')
    print("detection of : ", len(xMax), "   Validation of ==>  ", end='')
    #transform_show(fioOut)

    return lvaMaxima, lvaMinima



def get_extrem_bin(binArray, fio, lvaMinima):
    #for z, y, x in list(itertools.product(range(1,fio.shape[0]-1), range(1, fio.shape[1]-1), range(1,fio.shape[2]-1))):
    #    print(binArray[x,y,z])
    xpos, ypos, zpos = np.where(binArray==26)
    print(len(xpos))
    for i in range(len(xpos)):
        lvaMinima.append(Location_Value_XYZ(xpos[i], ypos[i], zpos[i], fio[xpos[i], ypos[i], zpos[i]]))
    return lvaMinima

# brut force
def ValleyFinderBrutForce(fio, es, lvaMinima):
    if fio.shape[2]>1:
        iZStart = 1
        iNeighbourIndexCount = 26
        iZCount = fio.shape[2]-1
        piNeighbourIndices = list(itertools.product([0,-1,1], repeat=3))
    else:
        iZStart = 0
        iNeighbourIndexCount = 8
        iZCount = 1
    for z, y, x in list(itertools.product(range(1,fio.shape[0]-1), range(1, fio.shape[1]-1), range(1,fio.shape[2]-1))):
        centerValue=fio[x, y, z]
        bPeak=True
        for xx, yy, zz in piNeighbourIndices[1:]:
            neighbord = fio[x+xx, y+yy, z+zz]
            bPeak=centerValue<neighbord
            if not bPeak:
                break
        if bPeak:
            if type(es.pfioH) is nibabel.nifti1.Nifti1Image:
                for xx, yy, zz in piNeighbourIndices:
                    neighbord = es.pfioH.get_fdata()[x+xx, y+yy, z+zz]
                    bPeak=centerValue<neighbord
                    if not bPeak:
                        break
            if type(es.pfioL) is nibabel.nifti1.Nifti1Image:
                for xx, yy, zz in piNeighbourIndices:
                    neighbord = es.pfioL.get_fdata()[x+xx, y+yy, z+zz]
                    bPeak=centerValue<neighbord
                    if not bPeak:
                        break
            if bPeak:
                lvaMinima.append(Location_Value_XYZ(x, y, z, centerValue))
    return lvaMinima

def PeakFinderBrutForce(fio, es, lvaMaxima):
    if fio.shape[2]>1:
        iZStart = 1
        iNeighbourIndexCount = 26
        iZCount = fio.shape[2]-1
        piNeighbourIndices = list(itertools.product([0,-1,1], repeat=3))
    else:
        iZStart = 0
        iNeighbourIndexCount = 8
        iZCount = 1
    for z, y, x in list(itertools.product(range(1,fio.shape[0]-1), range(1, fio.shape[1]-1), range(1,fio.shape[2]-1))):
        centerValue=fio[x, y, z]
        bPeak=True
        for xx, yy, zz in piNeighbourIndices[1:]:
            neighbord = fio[x+xx, y+yy, z+zz]
            bPeak=centerValue>neighbord
            if not bPeak:
                break
        if bPeak:
            if type(es.pfioH) is nibabel.nifti1.Nifti1Image:
                for xx, yy, zz in piNeighbourIndices:
                    neighbord = es.pfioH.get_fdata()[x+xx, y+yy, z+zz]
                    bPeak=centerValue>neighbord
                    if not bPeak:
                        break
            if bPeak:
                lvaMaxima.append(Location_Value_XYZ(x, y, z, centerValue))
    return lvaMaxima


def detect_local_minima(arr):
    neighborhood = morphology.generate_binary_structure(len(arr.shape),3)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    local_max = (filters.maximum_filter(arr, footprint=neighborhood)==arr)
    detected_minima = (local_min.astype(np.float32) - local_max.astype(np.float32))>0
    detected_minima = np.asarray(np.where(detected_minima))
    return detected_minima


def detect_local_minima_UP_OR_DOWN(arr, arr2, fio, lvaMinima):
    for i in range(len(arr2[0])):
        x = arr2[0, i]
        y = arr2[1, i]
        z = arr2[2, i]
        if x==arr.shape[0] or x==0 or y==arr.shape[1] or y==0 or z==arr.shape[2] or z==0:
            continue #error here, should not doing this we could lose feature with small size or large filter
        bPeak=arr[x-1:x+2,y-1:y+2,z-1:z+2]>fio[x, y, z]
        if len(np.where(bPeak==False)[0])==0:
            lvaMinima.append(Location_Value_XYZ(x,y, z, fio[x, y, z]))
    return lvaMinima

def detect_local_minimaUP_DOWN(arr, arr2, index, fio, lvaMinima):
    for i in range(len(arr2[0])):
        x = index[0, i]
        y = index[1, i]
        z = index[2, i]
        bPeak=np.min(arr[x-1:x+2,y-1:y+2,z-1:z+2])>fio[x, y, z] and np.min(arr2[x-1:x+2,y-1:y+2,z-1:z+2])>fio[x, y, z]
        if bPeak:
            lvaMinima.append(Location_Value_XYZ(x,y, z, fio[x, y, z]))
    return lvaMinima


def detect_local_maxima(arr):
    neighborhood = morphology.generate_binary_structure(len(arr.shape),3)
    local_max = (filters.maximum_filter(arr, footprint=neighborhood)==arr)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    detected_minima = (local_max.astype(np.float32) - local_min.astype(np.float32))>0
    return np.asarray(np.where(detected_minima))

def detect_local_extremum(arr, es, lvaMaxima, lvaMinima):
    piNeighbourIndices = list(itertools.product([0,-1,1], repeat=3))
    arr2=np.zeros(arr.shape)
    for xx, yy, zz in piNeighbourIndices[1:]:
        arr2[1: arr.shape[0]-1, 1: arr.shape[1]-1, 1: arr.shape[2]-1]+=np.sign(arr[1+xx: arr.shape[0]-1+xx, 1+yy: arr.shape[1]-1+yy, 1+zz: arr.shape[2]-1+zz]-arr[1: arr.shape[0]-1, 1: arr.shape[1]-1, 1: arr.shape[2]-1])
    for xx, yy, zz in piNeighbourIndices:
        arr2[1: arr.shape[0]-1, 1: arr.shape[1]-1, 1: arr.shape[2]-1]+=np.sign(es.pfioH.get_fdata()[1+xx: arr.shape[0]-1+xx, 1+yy: arr.shape[1]-1+yy, 1+zz: arr.shape[2]-1+zz]-arr[1: arr.shape[0]-1, 1: arr.shape[1]-1, 1: arr.shape[2]-1])
    x, y, z = np.asarray(np.where(arr2==-53))
    for i in range(len(x)):
        lvaMaxima.append(Location_Value_XYZ(x[i],y[i], z[i], arr[x[i], y[i], z[i]]))
    x, y, z = np.asarray(np.where(arr2==53))
    for i in range(len(x)):
        lvaMinima.append(Location_Value_XYZ(x[i],y[i], z[i], arr[x[i], y[i], z[i]]))
    return lvaMaxima, lvaMinima

"""def detect_local_extremumMax(arr, es, lvaMaxima):
    piNeighbourIndices = list(itertools.product([0,-1,1], repeat=3))
    arr2=np.zeros(arr.shape)
    end=26
    for xx, yy, zz in piNeighbourIndices[1:]:
        arr2[1: arr.shape[0]-1, 1: arr.shape[1]-1, 1: arr.shape[2]-1]+=np.sign(arr[1+xx: arr.shape[0]-1+xx, 1+yy: arr.shape[1]-1+yy, 1+zz: arr.shape[2]-1+zz]-arr[1: arr.shape[0]-1, 1: arr.shape[1]-1, 1: arr.shape[2]-1])
    if type(es.pfioH) is nibabel.nifti1.Nifti1Image:
        for xx, yy, zz in piNeighbourIndices:
            arr2[1: arr.shape[0]-1, 1: arr.shape[1]-1, 1: arr.shape[2]-1]+=np.sign(es.pfioH.get_fdata()[1+xx: arr.shape[0]-1+xx, 1+yy: arr.shape[1]-1+yy, 1+zz: arr.shape[2]-1+zz]-arr[1: arr.shape[0]-1, 1: arr.shape[1]-1, 1: arr.shape[2]-1])
        end=53
    x, y, z = np.asarray(np.where(arr2==-end))
    transform_showCPU(arr2)
    for i in range(len(x)):
        lvaMaxima.append(Location_Value_XYZ(x[i],y[i], z[i], arr[x[i], y[i], z[i]]))
    return lvaMaxima"""

def detect_local_extremumMax(fio, es, lvaMaxima):
    piNeighbourIndices = list(itertools.product([0,-1,1], repeat=3))
    for x, y, z in list(itertools.product(range(1,fio.shape[0]-1), range(1, fio.shape[1]-1), range(1,fio.shape[2]-1))):
        center=fio[x, y, z]
        bPeak=True
        for xx, yy, zz in piNeighbourIndices[1:]:
            if fio[x+xx, y+yy, z+zz]>=center:
                bPeak=False
                break
        if bPeak==True:
            lvaMaxima.append(Location_Value_XYZ(x,y, z, fio[x, y, z]))
    return lvaMaxima


def detect_local_maxima_UP_OR_DOWN(arr, arr2, fio, lvaMaxima):
    for i in range(len(arr2[0])):
        x = arr2[0, i]
        y = arr2[1, i]
        z = arr2[2, i]
        if x==arr.shape[0] or x==0 or y==arr.shape[1] or y==0 or z==arr.shape[2] or z==0:
            continue
        bPeak=arr[x-1:x+2,y-1:y+2,z-1:z+2]<fio[x, y, z]
        if len(np.where(bPeak==False)[0])==0:
            lvaMaxima.append(Location_Value_XYZ(x,y, z, fio[x, y, z]))
    return lvaMaxima

def detect_local_maximaUP_DOWN(arr, arr2, index, fio, lvaMaxima):
    for i in range(len(arr2[0])):
        x = index[0, i]
        y = index[1, i]
        z = index[2, i]
        bPeak=np.max(arr[x-1:x+2,y-1:y+2,z-1:z+2])<fio[x, y, z]
        if bPeak:
            lvaMaxima.append(Location_Value_XYZ(x,y, z, fio[x, y, z]))
    return lvaMaxima


def regFindFEATUREIOValleys(fio, es, lvaMinima):
    index=detect_local_minima(fio) # Quickly get current image index for extremum (min)
    if type(es.pfioH) is nibabel.nifti1.Nifti1Image and type(es.pfioL) is nibabel.nifti1.Nifti1Image:
        lvaMinima = detect_local_minimaUP_DOWN(es.pfioH.get_fdata(), es.pfioL.get_fdata(), index, fio, lvaMinima)
    elif type(es.pfioH) is nibabel.nifti1.Nifti1Image:
        lvaMinima=detect_local_minima_UP_OR_DOWN(es.pfioH.get_fdata(), index, fio, lvaMinima)
    elif type(es.pfioL) is nibabel.nifti1.Nifti1Image:
        lvaMinima=detect_local_minima_UP_OR_DOWN(es.pfioL.get_fdata(), index, fio, lvaMinima)
    else:
        print("Quick search output. \n Be careful these point could be wrong at low scale")
        for x, y, z in list(zip(index[0], index[1], index[2])):
            lvaMinima.append(Location_Value_XYZ(x, y, z, fio[x, y, z]))
    return lvaMinima

def regFindFEATUREIOPeak(fio, es, lvaMaxima):
    index=detect_local_maxima(fio) # Quickly get current image index for extremum (min)
    if type(es.pfioH) is nibabel.nifti1.Nifti1Image and type(es.pfioL) is nibabel.nifti1.Nifti1Image:
        lvaMaxima = detect_local_maximaUP_DOWN(es.pfioH.get_fdata(), es.pfioL.get_fdata(), index, fio, lvaMaxima)
    elif type(es.pfioH) is nibabel.nifti1.Nifti1Image:
        lvaMaxima=detect_local_maxima_UP_OR_DOWN(es.pfioH.get_fdata(), index, fio, lvaMaxima)
    elif type(es.pfioL) is nibabel.nifti1.Nifti1Image:
        lvaMaxima=detect_local_maxima_UP_OR_DOWN(es.pfioL.get_fdata(), index, fio, lvaMaxima)
    else:
        print("Quick search output. \n Be careful these point could be wrong at low scale")
        for x, y, z in list(zip(index[0], index[1], index[2])):
            lvaMaxima.append(Location_Value_XYZ(x, y, z, fio[x, y, z]))
    return lvaMaxima

def OLDregFindFEATUREIOValleys(fio, es, lvaMinima):
    if fio.shape[2]>1:
        iZStart = 1
        iNeighbourIndexCount = 26
        iZCount = fio.shape[2]-1
        piNeighbourIndices = list(itertools.product([0,-1,1], repeat=3))[1:]
    else:
        iZStart = 0
        iNeighbourIndexCount = 8
        iZCount = 1

    # Another method (probably shorter)
    # could be to sub 26 fio with shift and see for each where is max
    # but it will be an important memory cost.
    for z, y, x in list(itertools.product(range(1,fio.shape[0]-1), range(1, fio.shape[1]-1), range(1,fio.shape[2]-1))):
        bPeak=np.argmin(fio[x-1:x+2,y-1:y+2,z-1:z+2])==13
        if not bPeak:
            continue
        bPeak=np.min(es.pfioH[x-1:x+2,y-1:y+2,z-1:z+2])>np.min(fio[x-1:x+2,y-1:y+2,z-1:z+2])
        if not bPeak:
            continue
        if es.pfioL!=False:
            bPeak = np.min(es.pfioL[x-1:x+2,y-1:y+2,z-1:z+2])>np.min(fio[x-1:x+2,y-1:y+2,z-1:z+2])
            if not bPeak:
                continue
        #es.add_ext(iNeighbourIndexCount, x, y, z, fio.get_fdata()[x,y,z], piNeighbourIndices)
        lvaMinima.append(Location_Value_XYZ(x,y, z, fio[x,y,z]))
    return lvaMinima


def OLDregFindFEATUREIOPeaks(fio, es, lvaMaxima):
    if fio.shape[2]>1:
        iZStart = 1
        iNeighbourIndexCount = 26
        iZCount = fio.shape[2]-1
        piNeighbourIndices = list(itertools.product([0,-1,1], repeat=3))[1:]
    else:
        iZStart = 0
        iNeighbourIndexCount = 8
        iZCount = 1
    for z, y, x in list(itertools.product(range(1,fio.shape[0]-1), range(1, fio.shape[1]-1), range(1,fio.shape[2]-1))):
        bPeak=np.argmax(fio[x-1:x+2,y-1:y+2,z-1:z+2])==13
        if not bPeak:
            continue
        if es.pfioH!=False:
            bPeak=np.max(es.pfioH[x-1:x+2,y-1:y+2,z-1:z+2])<np.max(fio[x-1:x+2,y-1:y+2,z-1:z+2])
            if not bPeak:
                continue
        if es.pfioL!=False:
            bPeak = np.max(es.pfioL[x-1:x+2,y-1:y+2,z-1:z+2])<np.max(fio[x-1:x+2,y-1:y+2,z-1:z+2])
            if not bPeak:
                continue
        #es.add_ext(iNeighbourIndexCount, x, y, z, fio.get_fdata()[x,y,z], piNeighbourIndices)
        lvaMaxima.append(Location_Value_XYZ(x,y, z, fio[x,y,z]))
    return lvaMaxima


def validateDifferencePeak3D(fio, fio2, lvPeak):
    if fio.z>1:
        iZstart = 1
        iNeighbourIndexCount = 27
        iZCount = fio.z-1
        piNeighbourIndices=np.zeros(iNeighbourIndexCount)
        piNeighbourIndices[0] = 0
        piNeighbourIndices[1] = -fio.x*fio.y - fio.x - 1
        piNeighbourIndices[2] = -fio.x*fio.y - fio.x
        piNeighbourIndices[3] = -fio.x*fio.y - fio.x + 1
        piNeighbourIndices[4] = -fio.x*fio.y - 1
        piNeighbourIndices[5] = -fio.x*fio.y
        piNeighbourIndices[6] = -fio.x*fio.y + 1
        piNeighbourIndices[7] = -fio.x*fio.y + fio.x - 1
        piNeighbourIndices[8] = -fio.x*fio.y + fio.x
        piNeighbourIndices[9] = -fio.x*fio.y + fio.x + 1

        piNeighbourIndices[10] = -fio.x - 1
        piNeighbourIndices[11] = -fio.x
        piNeighbourIndices[12] = -fio.x + 1
        piNeighbourIndices[13] = -1
        piNeighbourIndices[14] = 1
        piNeighbourIndices[15] = fio.x - 1
        piNeighbourIndices[16] = fio.x
        piNeighbourIndices[17] = fio.x + 1

        piNeighbourIndices[18] = fio.x*fio.y - fio.x - 1
        piNeighbourIndices[19] = fio.x*fio.y - fio.x
        piNeighbourIndices[20] = fio.x*fio.y - fio.x + 1
        piNeighbourIndices[21] = fio.x*fio.y - 1
        piNeighbourIndices[22] = fio.x*fio.y
        piNeighbourIndices[23] = fio.x*fio.y + 1
        piNeighbourIndices[24] = fio.x*fio.y + fio.x - 1
        piNeighbourIndices[25] = fio.x*fio.y + fio.x
        piNeighbourIndices[26] = fio.x*fio.y + fio.x + 1
    else:
        iZStart = 0
        iNeighbourIndexCount = 8
        iZCount = 1
    fCenterValue = lvPeak.fValue
    iIndex=lvPeak.x+lvPeak.y*fio.x+lvPeak.z*fio.x*fio.y
    for n in range(len(piNeighbourIndices)):
        Index=int(iIndex+piNeighbourIndices[n])
        fcubic_data = fio._cpu_data[Index]-fio2._cpu_data[Index]
        if fcubic_data>=fCenterValue:
            return False
    return True


def validateDifferenceValley3D(fio, fio2, lvPeak):
    if fio.z>1:
        iZstart = 1
        iNeighbourIndexCount = 27
        iZCount = fio.z-1
        piNeighbourIndices=np.zeros(iNeighbourIndexCount)
        piNeighbourIndices[0] = 0
        piNeighbourIndices[1] = -fio.x*fio.y - fio.x - 1
        piNeighbourIndices[2] = -fio.x*fio.y - fio.x
        piNeighbourIndices[3] = -fio.x*fio.y - fio.x + 1
        piNeighbourIndices[4] = -fio.x*fio.y - 1
        piNeighbourIndices[5] = -fio.x*fio.y
        piNeighbourIndices[6] = -fio.x*fio.y + 1
        piNeighbourIndices[7] = -fio.x*fio.y + fio.x - 1
        piNeighbourIndices[8] = -fio.x*fio.y + fio.x
        piNeighbourIndices[9] = -fio.x*fio.y + fio.x + 1

        piNeighbourIndices[10] = -fio.x - 1
        piNeighbourIndices[11] = -fio.x
        piNeighbourIndices[12] = -fio.x + 1
        piNeighbourIndices[13] = -1
        piNeighbourIndices[14] = 1
        piNeighbourIndices[15] = fio.x - 1
        piNeighbourIndices[16] = fio.x
        piNeighbourIndices[17] = fio.x + 1

        piNeighbourIndices[18] = fio.x*fio.y - fio.x - 1
        piNeighbourIndices[19] = fio.x*fio.y - fio.x
        piNeighbourIndices[20] = fio.x*fio.y - fio.x + 1
        piNeighbourIndices[21] = fio.x*fio.y - 1
        piNeighbourIndices[22] = fio.x*fio.y
        piNeighbourIndices[23] = fio.x*fio.y + 1
        piNeighbourIndices[24] = fio.x*fio.y + fio.x - 1
        piNeighbourIndices[25] = fio.x*fio.y + fio.x
        piNeighbourIndices[26] = fio.x*fio.y + fio.x + 1
    else:
        iZStart = 0
        iNeighbourIndexCount = 8
        iZCount = 1
    fCenterValue = lvPeak.fValue
    iIndex=lvPeak.x+lvPeak.y*fio.x+lvPeak.z*fio.x*fio.y
    for n in range(len(piNeighbourIndices)):
        Index=int(iIndex+piNeighbourIndices[n])
        fcubic_data = fio._cpu_data[Index]-fio2._cpu_data[Index]
        if fcubic_data<=fCenterValue:
            return False
    return True


def print_img(img):
    nib.save(img, 'test3d.nii')

def dtoh_1dto3d(fio):
    fio._cpu_data=np.array(fio.x*fio.y*fio.z, dtype=np.ndarray)
    fio._cpu_data=fio.data.get()
    fio._cpu_data = fio._cpu_data.reshape((fio.x,fio.y,fio.z)).transpose()

# transform array to nii image and print it
def transform_show(fioIn):
    affine = np.asarray([[  1., 0., 0., 0 ],
             [   0., 1 , 0., 0 ],
             [   0., 0., 1., 0 ],
             [   0., 0., 0., 1.]])
    if type(fioIn) is sift.FEATUREIO:
        fioIn2=(fioIn.data.get()[:fioIn.x*fioIn.y*fioIn.z]).reshape((fioIn.x,fioIn.y,fioIn.z)).transpose()
        fioNi = nib.Nifti1Image(fioIn2, affine)
    else:
        fioNi = nib.Nifti1Image(fioIn, affine)
    show_image(fioNi)



def transform_showCPU(fioIn):
    affine = np.asarray([[  1., 0., 0., 0 ],
             [   0., 1 , 0., 0 ],
             [   0., 0., 1., 0 ],
             [   0., 0., 0., 1.]])
    if type(fioIn) is sift.FEATUREIO:
        #myout=np.array(fioIn.x*fioIn.y*fioIn.z, dtype=np.ndarray)
        #myout=fioIn.data.get()
        #fioIn = myout.reshape((fioIn.x,fioIn.y,fioIn.z)).transpose()
        fioIn2=(fioIn._cpu_data[:fioIn.x*fioIn.y*fioIn.z]).reshape((fioIn.x,fioIn.y,fioIn.z)).transpose()
        fioNi = nib.Nifti1Image(fioIn2, affine)
    else:
        fioNi = nib.Nifti1Image(fioIn, affine)
    show_image(fioNi)




def fioDoubleSize2( fioIn ):
    print("En cour de production")
    return None
    if (fioIn.header['dim'])[4] > 1:
        print("Doubling not implemeneted for multidimensional images.\n")
        raise SystemExit
    fioDouble = fioIn
    print(fioDouble.shape)
    print(fioDouble.header['dim'])
    if fioDouble.shape[0] > 1:
        newShape = (fioDouble.header['dim'][1] *2,fioDouble.header['dim'][2],fioDouble.header['dim'][3])
        fioDouble.header['dim'][1]=fioDouble.header['dim'][1]*2
        (fioDouble.header).set_data_shape(newShape)
    if fioDouble.shape[1] > 1:
        newShape = (fioDouble.header['dim'][1],fioDouble.header['dim'][2]*2,fioDouble.header['dim'][3])
        fioDouble.header['dim'][2]=fioDouble.header['dim'][2]*2
        (fioDouble.header).set_data_shape(newShape)
    if fioDouble.shape[2] > 1:
        newShape = (fioDouble.header['dim'][1],fioDouble.header['dim'][2],fioDouble.header['dim'][3]*2)
        fioDouble.header['dim'][3]=fioDouble.header['dim'][3]*2
        (fioDouble.header).set_data_shape(newShape)
    print(fioDouble.header['dim'])
    for z in range(fioIn.shape[2]):
          for y in range(fioIn.shape[1]):
              for x in range(fioIn.shape[0]):
                  pfLowRes=[[[]]]
                  for zz in range(2):
                      dz = zz
                      if z + zz >= fioIn.shape[2]:
                          dz=0
                      for yy in range(2):
                          dy = yy
                          if y + yy >= fioIn.shape[1]:
                              dy=0
                          for xx in range(2):
                              dx = xx
                              if x + xx >= fioIn.shape[0]:
                                  dx=0
                              pfLowRes[zz][yy][xx] = fiogetpixel( fioIn, x+dx, y+dy, z+dz)
                  if pfLowRes[0][0][0] != 0:
                      pfLowRes[0][0][0] = pfLowRes[0][0][0]

                  #Create high-res image
                  pfHighRes[0][0][0] = pfLowRes[0][0][0]

                  pfHighRes[1][0][0] = 0.5*(pfLowRes[0][0][0] + pfLowRes[1][0][0])
                  pfHighRes[0][1][0] = 0.5*(pfLowRes[0][0][0] + pfLowRes[0][1][0])
                  pfHighRes[0][0][1] = 0.5*(pfLowRes[0][0][0] + pfLowRes[0][0][1])

                  pfHighRes[1][1][0] = 0.25*(pfLowRes[0][0][0] + pfLowRes[1][0][0] + pfLowRes[0][1][0] + pfLowRes[1][1][0])
                  pfHighRes[0][1][1] = 0.25*(pfLowRes[0][0][0] + pfLowRes[0][1][0] + pfLowRes[0][0][1] + pfLowRes[0][1][1])
                  pfHighRes[1][0][1] = 0.25*(pfLowRes[0][0][0] + pfLowRes[1][0][0] + pfLowRes[0][0][1] + pfLowRes[1][0][1])

                  pfHighRes[1][1][1] = 0.125*(pfLowRes[0][0][0]
                                     + pfLowRes[0][0][1]
                                     + pfLowRes[0][1][0]
                                     + pfLowRes[0][1][1]
                                     + pfLowRes[1][0][0]
                                     + pfLowRes[1][0][1]
                                     + pfLowRes[1][1][0]
                                     + pfLowRes[1][1][1])
                  for zz in range(2):
                      dz = zz
                      if 2*z + zz >= fioDouble.header['dim'][3]:
                          dz=0
                      for yy in range(2):
                          dy = yy
                          if 2*y + yy >= fioDouble.header['dim'][2]:
                              dy=0
                          for xx in range(2):
                              dx = xx
                              if 2*x + xx >= fioDouble.header['dim'][1]:
                                  dx=0

def fioDoubleSize(fioIn):
    affine = fioIn.affine
    voxel_size = [(-1*affine[0][0])/2,(affine[1][1])/2,(affine[2][2])/2]
    resample_img = nibabel.processing.resample_to_output(fioIn, voxel_size)
    nibabel.save(resample_img, "outputpath.nii")
    fioIn = nibabel.load("outputpath.nii")
    #fioIn = resample_img(fioIn, target_affine = fioIn.affine, target_shape=(int(fioIn.shape[0]*2), int(fioIn.shape[1]*2), int(fioIn.shape[2]*2)))
    return fioIn



def generateFeatures3D_efficient(lvaMinima,lvaMaxima,vecMinH,vecMinL,vecMaxH,vecMaxL,fioC,fScaleH,fScaleC,fScaleL,fioImg,fScale,vecFeats,fEigThres):
    feat3D = Feature3D()

    featSample = np.zeros((feat3D.FEATURE_3D_DIM, feat3D.FEATURE_3D_DIM, feat3D.FEATURE_3D_DIM))
    bInterpolate = True
    for i in range(len(lvaMinima)):
        feat3D = Feature3D()
        if not bInterpolate:
            feat3D.add_coord_bInterpolate(lvaMinima[i].x, lvaMinima[i].y, lvaMinima[i].z, fScale)
        else:
            feat3D.interpolate_discrete_3D_point_flat(fioC,lvaMinima[i].x, lvaMinima[i].y, lvaMinima[i].z)

            feat3D.interpolate_scale(fScaleH,fScaleC,fScaleL,vecMinH[i], fioC._cpu_data[lvaMinima[i].x+lvaMinima[i].y*fioC.x+lvaMinima[i].z*fioC.x*fioC.y], vecMinL[i])
        feat3D.conv_loc_2_subpixel_prec(0.5)
        feat3D.m_uiInfo &= ~feat3D.INFO_FLAG_MIN0MAX1
        vecFeats = feat3D.generate_Feature3D(fioImg, fEigThres,vecFeats)
    for i in range(len(lvaMaxima)):
        feat3D = Feature3D()
        if not bInterpolate:
            feat3D.add_coord_bInterpolate(lvaMaxima[i].x, lvaMaxima[i].y, lvaMaxima[i].z, fScale)
        else:
            feat3D.interpolate_discrete_3D_point_flat(fioC,lvaMaxima[i].x, lvaMaxima[i].y, lvaMaxima[i].z)
            feat3D.interpolate_scale(fScaleH,fScaleC,fScaleL,vecMaxH[i], fioC._cpu_data[lvaMaxima[i].x+lvaMaxima[i].y*fioC.x+lvaMaxima[i].z*fioC.x*fioC.y], vecMaxL[i])
        feat3D.conv_loc_2_subpixel_prec(0.5)
        feat3D.m_uiInfo |= feat3D.INFO_FLAG_MIN0MAX1
        vecFeats = feat3D.generate_Feature3D(fioImg, fEigThres,vecFeats)
    print("LEN FEAT : ", len(vecFeats))
    return vecFeats

def vec3D_dot_3d(pfP1, pfP2):
    return pfP1[0] * pfP2[0] + pfP1[1] * pfP2[1] + pfP1[2] * pfP2[2]

def vec3D_norm_3d(pfP):
    sumSqr = float(pfP[0]*pfP[0] + pfP[1]*pfP[1] + pfP[2]*pfP[2])
    if sumSqr>0:
        fDiv=float(1.0/np.sqrt(sumSqr))
        pfP*=fDiv
    else:
        pfP[0]=1
        pfP[1]=0
        pfP[2]=0
    return pfP

def vec3D_cross_3d(pfP1, pfP2):
    pfP3=np.zeros(3)
    pfP3[0]=pfP1[1]*pfP2[2] - pfP1[2]*pfP2[1]
    pfP3[1]=-pfP1[0]*pfP2[2] + pfP1[2]*pfP2[0]
    pfP3[2]=pfP1[0]*pfP2[1] - pfP1[1]*pfP2[0]
    return pfP3





def fioSubSample2DCenterPixel( fioIn, fioOut ):
    return fioIn


"""
# OLD ONE USING AFFINE MATRIX
def fioSubSample(fioToSubSample):
    affine = fioToSubSample.affine
    i=1
    if affine[0][0]<0:
        i=-1
    voxel_size = [(i*affine[0][0])*2.0,(affine[1][1])*2.0,(affine[2][2])*2.0]
    resample_img = nibabel.processing.resample_to_output(fioToSubSample, voxel_size)
    nibabel.save(resample_img, "outputpath.nii")
    fioSubSampled = nibabel.load("outputpath.nii")
    return fioSubSampled
"""

def fioSubSample(fioToSubSample, fioSubSampled):
    if type(fioToSubSample) is nibabel.nifti1.Nifti1Image:
        fioNi = fioToSubSample.get_fdata()[:,:,:]
        fioOut=np.zeros((int(fioNi.shape[0]/2),int(fioNi.shape[1]/2),int(fioNi.shape[2]/2)))
        for z, y, x in list(itertools.product(range(0,int(fioNi.shape[0]/2)), range(0, int(fioNi.shape[1]/2)), range(0,int(fioNi.shape[2]/2)))):
            fioOut[x,y,z]=sum(fioNi[x*2:(x*2)+1,y*2:(y*2)+1,z*2:(z*2)+1])
        return nib.Nifti1Image(fioOut, fioToSubSample.affine)
    elif type(fioToSubSample) is sift.FEATUREIO:
        subsampleFunc=cuda.GpuFunc.get_function("cudaSubSampleInterpolate")

        tile_size=10
        cache_size=2*tile_size
        dimBlock=(int(tile_size), int(tile_size), int(tile_size))
        dimCache = int(cache_size*cache_size*cache_size*fioToSubSample.iFeaturesPerVector*4)
        dimGrid=((math.ceil(fioSubSampled.x/float(tile_size))), (math.ceil(fioSubSampled.y/float(tile_size))), (math.ceil(fioSubSampled.z/float(tile_size))))

        subsampleFunc(fioToSubSample.data, fioSubSampled.data, np.int32(fioToSubSample.x), np.int32(fioToSubSample.y), np.int32(fioToSubSample.z), np.int32(fioSubSampled.x), np.int32(fioSubSampled.y), np.int32(fioSubSampled.z), np.int32(fioSubSampled.iFeaturesPerVector), np.int32(tile_size), np.int32(cache_size), block=(tile_size,tile_size,tile_size), grid=dimGrid, shared=dimCache)
        fioSubSampled._cpu_data=fioSubSampled.data.get()[:fioSubSampled.x*fioSubSampled.y*fioSubSampled.z*fioSubSampled.iFeaturesPerVector]
        #fioSubSampled=dtoh_1dto3d(fioSubSampled)
        #myout=np.empty_like(fioIn._cpu_data)
        #myout=fioOut.data.get()
        #transform_show(myout.reshape((fioOut.x,fioOut.y,fioOut.z)).transpose())
        return fioSubSampled

def getcopy(fioIn):
    fioOut = FEATUREIO()
    fioOut.x=fioIn.x
    fioOut.y=fioIn.y
    fioOut.z=fioIn.z
    fioOut.iFeaturesPerVector=fioIn.iFeaturesPerVector
    fioOut._cpu_data=fioIn._cpu_data
    return fioOut


def show_slices(slices):
    """" Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()

def show_image(fioIn):
    imgdata = fioIn.get_fdata()
    slice_0 = imgdata[int(fioIn.shape[0]/2), :, :]
    slice_1 = imgdata[:, int(fioIn.shape[1]/2), :]
    slice_2 = imgdata[:, :, int(fioIn.shape[2]/2)]
    show_slices([slice_0, slice_1, slice_2])

def show_info(fio):
    print("\n\t", fio.x)
    print("\t", fio.y)
    print("\t", fio.z)
    print("\t", fio.iFeaturesPerVector, "\n")
    print(fio._cpu_data)



def msGeneratePyramidDOG3D_efficient(fioIn, vecFeats3D, fInitialBlurScale, bDense, fEigThres, witchCuda=1, bOutput=0):
    iScaleCount=4
    vecLVAMin=[] # list of location value (min)
    vecLVAMax=[] # list of location value (max)
    vecMinH=[]
    vecMinL=[]
    vecMaxH=[]
    vecMaxL=[]
    FioBlur=[]
    lvaMaxima=[]
    lvaMinima=[]
    for i in range(4): #store 4 extra image for blur pyramid
        fioImg=FEATUREIO()
        fioImg.x=fioIn.x
        fioImg.y=fioIn.y
        fioImg.z=fioIn.z
        fioImg.iFeaturesPerVector=fioIn.iFeaturesPerVector
        fioImg.data= fioIn.data.copy()
        fioImg._cpu_data= fioIn._cpu_data
        FioBlur.append(fioImg)

    #set fio temp for blur filter (automatically resize execept memory size)
    fioTemp=FEATUREIO()
    fioTemp.x=fioIn.x
    fioTemp.y=fioIn.y
    fioTemp.z=fioIn.z
    fioTemp.iFeaturesPerVector=fioIn.iFeaturesPerVector
    fioTemp.data= gpuarray.empty(fioTemp.x*fioTemp.y*fioTemp.z, dtype=np.float32)

    fioSaveHalf=FEATUREIO()
    fioSaveHalf.x=math.floor(fioIn.x/2)
    fioSaveHalf.y=math.floor(fioIn.y/2)
    fioSaveHalf.z=math.floor(fioIn.z/2)
    fioSaveHalf.iFeaturesPerVector=int(fioIn.iFeaturesPerVector)
    fioSaveHalf.data= gpuarray.empty(fioSaveHalf.x*fioSaveHalf.y*fioSaveHalf.z, dtype=np.float32)

    fSigmaInit = 0.5
    if fInitialBlurScale>0:
        fSigmaInit /= fInitialBlurScale
    fSigma = 1.6
    fSigmaFactor = math.pow(2.0,(1.0/BLURS_PER_OCTAVE))
    fSigmaExtra = math.sqrt(fSigma**2 - fSigmaInit**2)

    start = time.time()
    FioBlur[0], iReturn = gb3d_blur3d(fioIn, FioBlur[0], fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)
    end = time.time()
    if bOutput:
        transform_show(FioBlur[0])
    fScale = 1
    pfBlurSigmas = np.zeros(8)
    fioD0 = fioIn
    iscale=0
    nbfeatMax=0
    nbfeatMin=0


    fioG0=getcopy(fioIn)
    fioG1=getcopy(fioIn)
    fioG2=getcopy(fioIn)
    fioD0=getcopy(fioIn)
    fioD1=getcopy(fioIn)
    sumtot=0
    sumtotmin=0
    sumtotmax=0
    while(1):
        print("Scale " + str(iscale) + ": blur...")
        fSigma = 1.6
        pfBlurSigmas[0] = fSigma
        fioG0.data = FioBlur[0].data
        fioG1.data = FioBlur[1].data
        fioG2.data = FioBlur[2].data
        fioD0.data = FioBlur[3].data

        fioG0._cpu_data = FioBlur[0]._cpu_data
        fioG1._cpu_data = FioBlur[1]._cpu_data
        fioG2._cpu_data = FioBlur[2]._cpu_data
        fioD0._cpu_data = FioBlur[3]._cpu_data

        if fioG0.x <=2 or fioG0.y <=2 or fioG0.z <=2:
            break
        for j in range(1, BLURS_TOTAL):
            fSigmaExtra = fSigma*math.sqrt(math.pow(fSigmaFactor,2)-1.0)
            if j == 1:
                fioG1, iReturn= gb3d_blur3d(fioG0, fioG1, fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)
                fioD0 = fioMultSum(fioG0, fioG1, fioD0, -1.0)
                if bOutput:
                    print("DOG")
                    transform_show(fioD0)
            else:

                fioG2, iReturn = gb3d_blur3d(fioG1, fioG2, fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)

                if j==BLURS_PER_OCTAVE:
                    fioSaveHalf = fioSubSample(fioG2, fioSaveHalf)
                    if bOutput:
                        print("fioSaveHalf")
                        transform_show(fioSaveHalf)
                if j>=3:
                    pass

                    iCurrTop=0
                    vecMaxL = []
                    vecMinL = []

                    for k in range(len(lvaMaxima)):

                        if not validateDifferencePeak3D(fioG1, fioG2, vecLVAMax[iCurrTop]):
                            del vecMaxH[iCurrTop]
                            del vecLVAMax[iCurrTop]
                        else:
                            vecMaxL.append(fioG1._cpu_data[vecLVAMax[iCurrTop].x+vecLVAMax[iCurrTop].y*fioG1.x+vecLVAMax[iCurrTop].z*fioG1.x*fioG1.y] - fioG2._cpu_data[vecLVAMax[iCurrTop].x+vecLVAMax[iCurrTop].y*fioG2.x+vecLVAMax[iCurrTop].z*fioG2.x*fioG2.y])
                            iCurrTop+=1
                    iCurrTop=0
                    for k in range(len(lvaMinima)):
                        #print(lvaMinima[k].x, lvaMinima[k].y, lvaMinima[k].z, lvaMinima[k].fValue)
                        if not validateDifferenceValley3D(fioG1, fioG2, vecLVAMin[iCurrTop]):
                            #print("reject")
                            del vecMinH[iCurrTop]
                            del vecLVAMin[iCurrTop]
                        else:
                            vecMinL.append(fioG1._cpu_data[vecLVAMin[iCurrTop].x+vecLVAMin[iCurrTop].y*fioG1.x+vecLVAMin[iCurrTop].z*fioG1.x*fioG1.y] - fioG2._cpu_data[vecLVAMin[iCurrTop].x+vecLVAMin[iCurrTop].y*fioG2.x+vecLVAMin[iCurrTop].z*fioG2.x*fioG2.y])
                            iCurrTop+=1
                    lvaMaxima=vecLVAMax
                    lvaMinima = vecLVAMin
                    #print(len(lvaMaxima))
                    sumtotmin+=len(lvaMinima)
                    sumtotmax+=len(lvaMaxima)
                    print(len(lvaMinima))
                    print(len(lvaMaxima))
                    nbfeatMax+=len(lvaMaxima)
                    nbfeatMin+=len(lvaMinima)

                    #print("Max : ",nbfeatMax)
                    #print("Min : ",nbfeatMin)
                    start = time.time()
                    print(pfBlurSigmas[j - 3], pfBlurSigmas[j - 2], pfBlurSigmas[j - 1], pfBlurSigmas[j + 1])
                    #vecFeats3D = generateFeatures3D_efficient(lvaMinima,lvaMaxima,vecMinH,vecMinL,vecMaxH,vecMaxL,fioD0,pfBlurSigmas[j - 3], pfBlurSigmas[j - 2], pfBlurSigmas[j - 1],fioG0,pfBlurSigmas[j + 1],vecFeats3D,fEigThres)
                    end = time.time()
                    print(end-start)

                if j < BLURS_TOTAL-1:
                    lvaMaxima=[]
                    lvaMinima=[]
                    vecMaxH=[]
                    vecMinH=[]
                    fioD1.data=fioG0.data
                    fioD1._cpu_data=fioG0._cpu_data
                    fioD1 = fioMultSum(fioG1, fioG2, fioD1, -1.0)
                    lvaMinima, lvaMaxima = detectExtrema4D_test(fioD0, fioD1, None, lvaMinima, lvaMaxima)

                    vecLVAMax = lvaMaxima
                    vecLVAMin = lvaMinima

                    #save value for scale below (scale upon will be check after)
                    for i in range(len(lvaMaxima)):
                        vecMaxH.append(fioD0._cpu_data[vecLVAMax[i].x+vecLVAMax[i].y*fioD0.x + vecLVAMax[i].z*fioD0.x*fioD0.y])
                    for i in range(len(lvaMinima)):
                        vecMinH.append(fioD0._cpu_data[vecLVAMin[i].x+vecLVAMin[i].y*fioD0.x + vecLVAMin[i].z*fioD0.x*fioD0.y])

                    pfG2Temp = fioG2.data
                    fioG2.data = fioD0.data
                    fioD0.data = fioD1.data
                    fioG0.data = fioG1.data
                    fioG1.data = pfG2Temp
                    pfG2Temp = fioG2._cpu_data
                    fioG2._cpu_data = fioD0._cpu_data
                    fioD0._cpu_data = fioD1._cpu_data
                    fioG0._cpu_data = fioG1._cpu_data
                    fioG1._cpu_data = pfG2Temp

            fSigma *= fSigmaFactor
            pfBlurSigmas[j] = fSigma
            fScale *= 2.0

        FioBlur[0].copy(fioSaveHalf)
        """FioBlur[1].copy(fioSaveHalf)
        FioBlur[2].copy(fioSaveHalf)
        FioBlur[3].copy(fioSaveHalf)"""
        fioG0.copy(FioBlur[0])
        fioG1.copy(FioBlur[0])
        fioG2.copy(FioBlur[0])
        fioD0.copy(FioBlur[0])
        fioD1.copy(FioBlur[0])

        FioBlur[0].data = fioSaveHalf.data.copy()

        fioSaveHalf.x=math.floor(fioSaveHalf.x/2)
        fioSaveHalf.y=math.floor(fioSaveHalf.y/2)
        fioSaveHalf.z=math.floor(fioSaveHalf.z/2)
        #fioSubSampled.data.gpudata.free()
        #fioSubSampled.data= gpuarray.zeros(fioSubSampled.x*fioSubSampled.y*fioSubSampled.z, dtype=np.float32)
        iscale+=1
    sumtot = sumtotmax+sumtotmin
    print(sumtotmin)
    print(sumtotmax)
    print(sumtot)
    print(len(vecFeats3D))
    return vecFeats3D


# WARNING UNSTABLE FUNCTION (do the full 4d peak/valley detection without validation but only for scale 0...yet)
def msGeneratePyramidDOG3D_efficient_test(fioIn, vecFeats3D, fInitialBlurScale, bDense, fEigThres, witchCuda=1, bOutput=0):
    iScaleCount=4
    vecLVAMin=[] # list of location value (min)
    vecLVAMax=[] # list of location value (max)
    vecMinH=[]
    vecMinL=[]
    vecMaxH=[]
    vecMaxL=[]
    FioBlur=[]
    lvaMaxima=[]
    lvaMinima=[]
    for i in range(7): #store 4 extra image for blur pyramid
        fioImg=FEATUREIO()
        fioImg.x=fioIn.x
        fioImg.y=fioIn.y
        fioImg.z=fioIn.z
        fioImg.iFeaturesPerVector=fioIn.iFeaturesPerVector
        fioImg.data= fioIn.data.copy()
        fioImg._cpu_data= fioIn._cpu_data
        FioBlur.append(fioImg)

    #set fio temp for blur filter (automatically resize execept memory size)
    fioTemp=FEATUREIO()
    fioTemp.x=fioIn.x
    fioTemp.y=fioIn.y
    fioTemp.z=fioIn.z
    fioTemp.iFeaturesPerVector=fioIn.iFeaturesPerVector
    fioTemp.data= gpuarray.empty(fioTemp.x*fioTemp.y*fioTemp.z, dtype=np.float32)

    fioSaveHalf=FEATUREIO()
    fioSaveHalf.x=math.floor(fioIn.x/2)
    fioSaveHalf.y=math.floor(fioIn.y/2)
    fioSaveHalf.z=math.floor(fioIn.z/2)
    fioSaveHalf.iFeaturesPerVector=int(fioIn.iFeaturesPerVector)
    fioSaveHalf.data= FioBlur[1].data

    fSigmaInit = 0.5
    if fInitialBlurScale>0:
        fSigmaInit /= fInitialBlurScale
    fSigma = 1.6
    fSigmaFactor = math.pow(2.0,(1.0/BLURS_PER_OCTAVE))
    fSigmaExtra = math.sqrt(fSigma**2 - fSigmaInit**2)

    start = time.time()
    FioBlur[0], iReturn = gb3d_blur3d(fioIn, FioBlur[0], fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)
    end = time.time()
    if bOutput:
        transform_show(FioBlur[0])
    fScale = 1
    pfBlurSigmas = np.zeros(8)
    fioD0 = fioIn
    iscale=0
    nbfeatMax=0
    nbfeatMin=0


    fioG0=getcopy(fioIn)
    fioG1=getcopy(fioIn)
    fioG2=getcopy(fioIn)
    fioG3=getcopy(fioIn)
    fioD0=getcopy(fioIn)
    fioD1=getcopy(fioIn)
    fioD2=getcopy(fioIn)
    sumtot=0
    sumtotmin=0
    sumtotmax=0
    while(1):
        print("Scale " + str(iscale) + ": blur...")
        fSigma = 1.6
        pfBlurSigmas[0] = fSigma
        fSigmaExtra = math.sqrt(fSigma**2 - fSigmaInit**2)
        #FioBlur[0], iReturn = gb3d_blur3d(FioBlur[1], FioBlur[0], fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)
        fioG0.data = FioBlur[0].data
        fioG1.data = FioBlur[1].data
        fioG2.data = FioBlur[2].data
        fioG3.data = FioBlur[3].data
        fioD0.data = FioBlur[4].data
        fioD1.data = FioBlur[5].data
        fioD2.data = FioBlur[6].data
        #fioSaveHalf.data= fioG0.data

        fioG0._cpu_data = FioBlur[0]._cpu_data
        fioG1._cpu_data = FioBlur[1]._cpu_data
        fioG2._cpu_data = FioBlur[2]._cpu_data
        fioG3._cpu_data = FioBlur[3]._cpu_data
        fioD0._cpu_data = FioBlur[4]._cpu_data
        fioD1._cpu_data = FioBlur[5]._cpu_data
        fioD2._cpu_data = FioBlur[6]._cpu_data

        if fioG0.x <=2 or fioG0.y <=2 or fioG0.z <=2:
            break

        fSigmaExtra = fSigma*math.sqrt(math.pow(fSigmaFactor,2)-1.0)
        fioG1, iReturn= gb3d_blur3d(fioG0, fioG1, fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)
        fSigma *= fSigmaFactor
        pfBlurSigmas[1] = fSigma
        fSigmaExtra = fSigma*math.sqrt(math.pow(fSigmaFactor,2)-1.0)
        fioG2, iReturn = gb3d_blur3d(fioG1, fioG2, fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)
        fSigma *= fSigmaFactor
        pfBlurSigmas[2] = fSigma
        fSigmaExtra = fSigma*math.sqrt(math.pow(fSigmaFactor,2)-1.0)
        fioG3, iReturn = gb3d_blur3d(fioG2, fioG3, fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)

        fioD0 = fioMultSum(fioG0, fioG1, fioD0, -1.0)
        fioD1 = fioMultSum(fioG1, fioG2, fioD1, -1.0)
        fioD2 = fioMultSum(fioG2, fioG3, fioD2, -1.0)


        lvaMinima, lvaMaxima = detectExtrema4D_test(fioD0, fioD1, fioD2, lvaMinima, lvaMaxima)

        fSigma *= fSigmaFactor
        fSigmaExtra = fSigma*math.sqrt(math.pow(fSigmaFactor,2)-1.0)
        fioG1, iReturn= gb3d_blur3d(fioG3, fioG1, fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)
        fioD0 = fioMultSum(fioG3, fioG1, fioD0, -1.0)

        lvaMinima, lvaMaxima = detectExtrema4D_test(fioD1, fioD2, fioD0, lvaMinima, lvaMaxima)

        fSigma *= fSigmaFactor
        fSigmaExtra = fSigma*math.sqrt(math.pow(fSigmaFactor,2)-1.0)
        fioG2, iReturn= gb3d_blur3d(fioG1, fioG2, fSigmaExtra, BLUR_PRECISION, witchCuda, fioTemp)
        fioD1 = fioMultSum(fioG1, fioG2, fioD1, -1.0)

        lvaMinima, lvaMaxima = detectExtrema4D_test(fioD2, fioD0, fioD1, lvaMinima, lvaMaxima)

        #vecFeats3D = generateFeatures3D_efficient(lvaMinima,lvaMaxima,vecMinH,vecMinL,vecMaxH,vecMaxL,fioD0,pfBlurSigmas[0], pfBlurSigmas[1], pfBlurSigmas[2],fioG0,pfBlurSigmas[4],vecFeats3D,fEigThres)

        fioSaveHalf = fioSubSample(FioBlur[0], fioSaveHalf)

        fScale *= 2.0

        for i in range(7):
            FioBlur[i].copy(fioSaveHalf)
        fioG0.copy(FioBlur[0])
        fioG1.copy(FioBlur[0])
        fioG2.copy(FioBlur[0])
        fioG3.copy(FioBlur[0])
        fioD0.copy(FioBlur[0])
        fioD1.copy(FioBlur[0])
        fioD2.copy(FioBlur[0])
        fioTemp.copy(FioBlur[0])

        FioBlur[0].data = fioSaveHalf.data.copy()

        fioSaveHalf.x=math.floor(fioSaveHalf.x/2)
        fioSaveHalf.y=math.floor(fioSaveHalf.y/2)
        fioSaveHalf.z=math.floor(fioSaveHalf.z/2)
        iscale+=1
    print(len(vecFeats3D))
    return vecFeats3D

def msNormalizeDataPositive(Feat_m_pfPC, iLength):
    fMin=min(Feat_m_pfPC)
    Feat_m_pfPC -= fMin
    fSumSqr=sum(Feat_m_pfPC*Feat_m_pfPC)
    fDiv=1.0/float(np.sqrt(fSumSqr))
    fSumSqr=0
    Feat_m_pfPC*=fDiv
    return Feat_m_pfPC

def vec3D_mag(pf1):
    fSumSqr=pf1[0]*pf1[0] + pf1[1]*pf1[1] + pf1[2]*pf1[2]
    if fSumSqr>0:
        return np.sqrt(fSumSqr)
    else:
        return 0

def msResampleFeaturesGradientOrientationHistogram(Feature):
    fioDx, fioDy, fioDz = sift.fioGenerateEdgeImage3D(Feature.data_zyx)

    fRadius = float(Feature.FEATURE_3D_DIM/2)
    fRadiusSqr = float((Feature.FEATURE_3D_DIM/2)*(Feature.FEATURE_3D_DIM/2))
    iSampleCount=0
    fOriAngles=list(itertools.product((1,-1), repeat=3))
    fBinSize=Feature.FEATURE_3D_DIM/float(2)
    fioImgGradOri=np.zeros((2*8,2*8,2*8))
    assert(Feature.FEATURE_3D_PCS >= 64)
    for xx, yy, zz in list(itertools.product(range(Feature.FEATURE_3D_DIM), repeat=3)):

        fZCoord=(int(zz/fBinSize)+0.5)
        if (int((zz + 0) / fBinSize) != int((zz + 1) / fBinSize)):
            fP0=float((zz+0)/fBinSize)
            fP1=float((zz+1)/fBinSize)
            fZCoord=(fP0 + fP1) / 2.0

        fYCoord = float(int(yy / fBinSize) + 0.5)
        if (int((yy + 0) / fBinSize) != int((yy + 1) / fBinSize)):
            fP0=float((yy+0)/fBinSize)
            fP1=float((yy+1)/fBinSize)
            fYCoord=(fP0 + fP1) / 2.0

        fXCoord = float(int(xx / fBinSize) + 0.5)
        if (int((xx + 0) / fBinSize) != int((xx + 1) / fBinSize)):
            fP0=float((xx+0)/fBinSize)
            fP1=float((xx+1)/fBinSize)
            fXCoord=(fP0 + fP1) / 2.0
        dz=float(zz-Feature.FEATURE_3D_DIM/2)
        dy=float(yy-Feature.FEATURE_3D_DIM/2)
        dx=float(xx-Feature.FEATURE_3D_DIM/2)
        fEdge = np.zeros(3)
        fEdge[0]=fioDx[xx, yy, zz]
        fEdge[1]=fioDy[xx, yy, zz]
        fEdge[2]=fioDz[xx, yy, zz]
        fEdgeMag = vec3D_mag(fEdge)
        if fEdgeMag>0:
            fEdge=vec3D_norm_3d(fEdge)
            iMaxDotIndex=0
            fMaxDot=vec3D_dot_3d(fOriAngles[iMaxDotIndex], fEdge)
            for k in range(1,8):
                fDot = float(vec3D_dot_3d(fOriAngles[k], fEdge))
                if fDot>fMaxDot:
                    fMaxDot=fDot
                    iMaxDotIndex=k
            xyz=[fZCoord, fYCoord, fXCoord]

            Feature.m_pfPC=sift.fioIncPixelTrilinearInterp2(Feature.m_pfPC, xyz, fEdgeMag, 2, 2, 2, 8, iMaxDotIndex)

    Feature.m_pfPC=msNormalizeDataPositive(Feature.m_pfPC, Feature.FEATURE_3D_PCS)

    return Feature

def msFeature3DVectorOutputText(vecFeat, fileName, fEigThres, ppcCommentLines):
    outfile = open(fileName, "w")
    iFeat=0
    for feat in vecFeat:
        fEigSum = np.sum(feat.eigs)
        fEigPrd = np.prod(feat.eigs)
        fEigSumProd = fEigSum*fEigSum*fEigSum
        if fEigSumProd<fEigThres*fEigPrd or fEigThres<0:
            iFeat+=1
    outfile.write("# featExtract 1.1\n")
    for comment in ppcCommentLines:
        outfile.write(comment)
    outfile.write("Features: "+str(iFeat) + "\n")
    outfile.write("Scale-space location[x y z scale] orientation[o11 o12 o13 o21 o22 o23 o31 o32 o32] 2nd moment eigenvalues[e1 e2 e3] info flag[i1] descriptor[d1 .. d64]\n")
    for feat in vecFeat:
        fEigSum = np.sum(feat.eigs)
        fEigPrd = np.prod(feat.eigs)
        fEigSumProd = fEigSum*fEigSum*fEigSum
        if fEigSumProd<fEigThres*fEigPrd or fEigThres<0:
            pass
        else:
            continue
        outfile.write(str(round(feat.x, 6)) + "\t" + str(round(feat.y, 6)) + "\t"  + str(round(feat.z,6)) + "\t" + str(round(feat.scale, 6)) + "\t", )
        for j, k in list(itertools.product(range(3), repeat=2)):
            outfile.write(str(round(feat.ori[j,k], 6)) + "\t")
        for j in range(3):
            outfile.write(str(round(feat.eigs[j], 6)) + "\t")
        outfile.write(str(feat.m_uiInfo) + "\t")
        for j in range(feat.FEATURE_3D_PCS):
            outfile.write(str(int(feat.m_pfPC[j])) + "\t")
        outfile.write("\n")
    outfile.close()

# Let python control everything just use it as numpy would (get is for gpu->cpu)
def fioMultSumOLD(fioG0, fioG1, fioD, coeff):
    fioD.data=fioG0.data+(fioG1.data*float(coeff))
    fioD._cpu_data=fioD.data.get()
    return fioD

# Or Set parameter and get better perf
def fioMultSum(fioG0, fioG1, fioD, coeff):
    dOGfunc=cuda.GpuFunc.get_function("CudaMultSum")
    tile_size=10
    dimBlock=(int(tile_size), int(tile_size), int(tile_size))
    dimGrid=((math.ceil(fioD.x/float(tile_size))), (math.ceil(fioD.y/float(tile_size))), (math.ceil(fioD.z/float(tile_size))))
    dOGfunc(fioG0.data, fioG1.data, fioD.data, np.float32(coeff), np.int32(fioD.x), np.int32(fioD.y), np.int32(fioD.z),block=(tile_size,tile_size,tile_size), grid=dimGrid)
    fioD._cpu_data=fioD.data.get()
    return fioD






######################### CONVOLUTION #########################

def generate_gaussian_filter1d(fSigmaCol, iCols):
    fMeanCol = int(iCols/2)
    assert fMeanCol >= 0.0 and fMeanCol < iCols
    fSigmaColSqr = math.pow(fSigmaCol,2)
    fScale = float(1.0/(fSigmaCol*math.sqrt(2.0*math.pi)))
    ImgFilter=fScale*np.exp((np.power((np.array(range(0,iCols))-fMeanCol), 2)/fSigmaColSqr)/float(-2.0))
    return ImgFilter

def calculate_gaussian_filter_size(fSigma, fMinValue):
    fPower = 0.0
    fValue = float(math.exp(fPower))
    fMaxValue = fValue
    i=0
    if fSigma==0:
        return 1
    fCurVolume = 1.0
    fNewVolume = 1.0
    while True:
        i+=1
        fCurVolume = fNewVolume
        fPower = float(i*i) / (-2.0*math.pow(fSigma, 2))
        fNewVolume = fCurVolume + 2*float(math.exp( fPower ))
        if fNewVolume - fCurVolume < 0.00001:
            break
    i=1
    while True:
        if fValue > fCurVolume*(1.0-fMinValue):
            break
        fPower = float(i**2) / float(-2.0*fSigma*fSigma)
        fValue += 2*float(math.exp(fPower))
        i+=1
    i-=1
    return 2*i+1

def gb3d_blur3d(fioIn, fioOut, fSigma, fMinValue, witchCuda, fioTemp=None):
    return gb3d_blur3d_interleave(fioIn, fioOut, fioTemp, 0, fSigma, fMinValue, witchCuda)


def gb3d_blur3d_interleave(fioIn, fioOut, fioTemp, iFeature, fSigma, fMinValue, witchCuda):
    assert fSigma >= 0.0
    assert fMinValue >= 0 and fMinValue < 1.0
    iCols = calculate_gaussian_filter_size(fSigma, fMinValue)
    assert iCols%2 == 1
    if fSigma > 0.0:
        ImgFilter = generate_gaussian_filter1d(fSigma, iCols)
    else:
        assert iCols == 1
        ImgFilter = [1,1]

    # Normalise filter
    ImgFilter=ImgFilter/np.sum(ImgFilter)

    if witchCuda:
        fioTemp.x=fioIn.x
        fioTemp.y=fioIn.y
        fioTemp.z=fioIn.z
    if witchCuda:
        return blur_3_x_1d_simpleborders_CUDA(fioIn, fioOut, fioTemp,iFeature, ImgFilter)
        #return blur_3d_simpleborders(fioIn, iFeature, ImgFilter)
    else:
        return blur_3d_simpleborders(fioIn, iFeature, ImgFilter)


# Simple 3d convolution
def blur_3d_simpleborders(fioIn, iFeature, ImgFilter):
    imageout = fioIn
    ndimage.convolve1d(imageout, ImgFilter, 0, imageout, mode='constant')
    ndimage.convolve1d(imageout, ImgFilter, 1, imageout, mode='constant')
    ndimage.convolve1d(imageout, ImgFilter, 2, imageout, mode='constant')
    return imageout, 0

def blur_1_x_3d_simpleborders_CUDA(fioIn, iFeature, ImgFilter):
    # Allocate data on device
    stream = cuda.stream()
    dataIn = fioIn.get_fdata().copy()
    dataOut = fioIn.get_fdata().copy()
    d_dataIn = cuda.to_device(dataIn, stream=stream)
    d_dataOut = cuda.to_device(dataOut, stream=stream)
    d_Filter = cuda.to_device(np.array(ImgFilter), stream=stream)

    # Set GPU Property
    kernel_size = len(ImgFilter)
    kernel_radius = int(kernel_size/2)
    tile_size = 10
    threads = (tile_size,tile_size,tile_size)
    blockX = math.ceil(fioIn.shape[0] / tile_size)
    blockY = math.ceil(fioIn.shape[1] / tile_size)
    blockZ = math.ceil(fioIn.shape[2] / tile_size)
    blocks = (blockX,blockY,blockZ)
    cache_size = (tile_size + (kernel_radius*2))
    cudaConvolve3d[blocks, threads, stream, (cache_size*cache_size*cache_size*4)](d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,fioIn.shape[0],fioIn.shape[1],fioIn.shape[2])

    fioOut = d_dataOut.copy_to_host(stream=stream)
    stream.synchronize()
    return nib.Nifti1Image(fioOut, fioIn.affine, fioIn.header), 0



##### cuda 3*1d convolution using the gputools package (2*faster than cpu) #####
def blur_3_x_1d_simpleborders_CUDA(fioIn, fioOut, fioTemp, iFeature, ImgFilter):
    if type(fioIn) is nibabel.nifti1.Nifti1Image:
        return nib.Nifti1Image(gputools.convolve_sep3(fioIn.get_fdata()[:,:,:], ImgFilter,ImgFilter,ImgFilter),fioIn.affine, fioIn.header), 0
    elif type(fioIn) is sift.FEATUREIO:
        rowfunc=cuda.GpuFunc.get_function("conv3d_shared_Row")
        colfunc=cuda.GpuFunc.get_function("conv3d_shared_Col")
        depthfunc=cuda.GpuFunc.get_function("conv3d_shared_Depth")
        kernel_size=len(ImgFilter)
        kernel_radius=int(len(ImgFilter)/2)
        tile_size=10
        cache_size=tile_size + (kernel_radius * 2)
        dimBlock=(int(tile_size), int(tile_size), int(tile_size))
        dimCache = int(cache_size*tile_size*tile_size*4)
        # send filter to GPU


        ImgFilter=ImgFilter.astype(np.float32)
        gpuFilter=pycuda.driver.mem_alloc(ImgFilter.nbytes)
        pycuda.driver.memcpy_htod(gpuFilter, ImgFilter)

        dimGrid=((math.ceil(fioIn.x/float(tile_size))), (math.ceil(fioIn.y/float(tile_size))), (math.ceil(fioIn.z/float(tile_size))))
        rowfunc(fioIn.data, fioOut.data, gpuFilter, np.int32(kernel_size), np.int32(kernel_radius), np.int32(cache_size), np.int32(fioIn.x), np.int32(fioIn.y), np.int32(fioIn.z), np.int32(fioOut.x), np.int32(fioOut.y), np.int32(fioOut.z), block=(tile_size,tile_size,tile_size), grid=dimGrid, shared=dimCache)

        dimGrid=((math.ceil(fioIn.y/float(tile_size))), (math.ceil(fioIn.z/float(tile_size))), (math.ceil(fioIn.x/float(tile_size))))
        colfunc(fioOut.data, fioTemp.data, gpuFilter, np.int32(kernel_size), np.int32(kernel_radius), np.int32(cache_size), np.int32(fioIn.x), np.int32(fioIn.y), np.int32(fioIn.z), np.int32(fioOut.x), np.int32(fioOut.y), np.int32(fioOut.z), block=(tile_size,tile_size,tile_size), grid=dimGrid, shared=dimCache)

        dimGrid=((math.ceil(fioIn.z/float(tile_size))), (math.ceil(fioIn.x/float(tile_size))), (math.ceil(fioIn.y/float(tile_size))))
        depthfunc(fioTemp.data, fioOut.data, gpuFilter, np.int32(kernel_size), np.int32(kernel_radius), np.int32(cache_size), np.int32(fioIn.x), np.int32(fioIn.y), np.int32(fioIn.z), np.int32(fioOut.x), np.int32(fioOut.y), np.int32(fioOut.z), block=(tile_size,tile_size,tile_size), grid=dimGrid, shared=dimCache)

        fioOut._cpu_data=fioOut.data.get()
        #myout=np.empty_like(fioIn._cpu_data)
        #myout=fioOut.data.get()
        #transform_show(myout.reshape((fioOut.x,fioOut.y,fioOut.z)).transpose())
        return fioOut, 0
    else:
        return gputools.convolve_sep3(fioIn, ImgFilter,ImgFilter,ImgFilter), 0



##### cuda 3*1d convolution without rotation using the jit package (2*slower than cpu) #####
"""def blur_3_x_1d_simpleborders_CUDA_jiit(fioIn, iFeature, ImgFilter):
    # Allocate data on device
    stream = cuda.stream()
    dataIn = fioIn.get_fdata().copy()
    dataOut = fioIn.get_fdata().copy()
    d_dataIn = cuda.to_device(dataIn, stream=stream)
    d_dataOut = cuda.to_device(dataOut, stream=stream)
    d_Filter = cuda.to_device(np.array(ImgFilter), stream=stream)

    # Set GPU Property
    kernel_size = len(ImgFilter)
    kernel_radius = int(kernel_size/2)
    tile_size = 10
    threads = (tile_size,tile_size,tile_size)
    blockX = math.ceil(fioIn.shape[0] / tile_size)
    blockY = math.ceil(fioIn.shape[1] / tile_size)
    blockZ = math.ceil(fioIn.shape[2] / tile_size)
    blocks = (blockX,blockY,blockZ)
    cache_size = (tile_size + (kernel_radius*2))
    dimcache = cache_size*tile_size*tile_size*4
    cudaConvolve1d_Row[blocks, threads, stream, dimcache](d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,fioIn.shape[0],fioIn.shape[1],fioIn.shape[2])
    cudaConvolve1d_Col[blocks, threads, stream, dimcache](d_dataOut,d_dataIn,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,fioIn.shape[0],fioIn.shape[1],fioIn.shape[2])
    cudaConvolve1d_Depth[blocks, threads, stream, dimcache](d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,fioIn.shape[0],fioIn.shape[1],fioIn.shape[2])

    fioOut = d_dataOut.copy_to_host(stream=stream)
    stream.synchronize()
    return nib.Nifti1Image(fioOut, fioIn.affine, fioIn.header), 0

@cuda.jit
def cudaConvolve1d_Row(d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,size_x,size_y,size_z):
    shared_data = cuda.shared.array(shape=0,dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    bx = cuda.blockIdx.x * tile_size
    by = cuda.blockIdx.y * tile_size
    bz = cuda.blockIdx.z * tile_size

    x_pos = bx + tx
    y_pos = by + ty
    z_pos = bz + tz
    tile_id = tz * cache_size * tile_size + ty * tile_size + tx
    if tile_id < tile_size*tile_size:
        tileZ = int(float(tile_id)/tile_size) % tile_size
        tileY = tile_id % tile_size
        input_z = bz + tileZ
        input_y = by + tileY
        input_x_root = bx - kernel_radius

        for stemLength in range(0,cache_size):
            input_x = input_x_root+stemLength
            if is_pixel(input_x,input_y,input_z,size_x,size_y,size_z):
                shared_data[tileZ*cache_size*tile_size + tileY*cache_size + stemLength] = d_dataIn[input_x, input_y, input_z]
            else:
                shared_data[tileZ*cache_size*tile_size + tileY*cache_size + stemLength] = 0.0
    cuda.syncthreads()
    if is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z):
        OutputValue = 0.0
        for x in range(0,kernel_size):
            OutputValue += shared_data[(tx+x)+(ty)*cache_size+(tz)*cache_size*tile_size]*d_Filter[x]
        d_dataOut[x_pos, y_pos, z_pos] = OutputValue

@cuda.jit
def cudaConvolve1d_Col(d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,size_x,size_y,size_z):
    shared_data = cuda.shared.array(shape=0,dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    bx = cuda.blockIdx.x * tile_size
    by = cuda.blockIdx.y * tile_size
    bz = cuda.blockIdx.z * tile_size

    x_pos = bx + tx
    y_pos = by + ty
    z_pos = bz + tz
    tile_id = tz * tile_size * tile_size + ty * tile_size + tx
    if tile_id < tile_size*tile_size:
        tileY = int(float(tile_id)/tile_size) % tile_size
        tileX = tile_id % tile_size
        input_x = bx + tileX
        input_y = by + tileY
        input_z_root = bz - kernel_radius

        for stemLength in range(0,cache_size):
            input_z = input_z_root+stemLength
            if is_pixel(input_x,input_y,input_z,size_x,size_y,size_z):
                shared_data[stemLength*tile_size*tile_size + tileY*tile_size + tileX] = d_dataIn[input_x, input_y, input_z]
            else:
                shared_data[stemLength*tile_size*tile_size + tileY*tile_size + tileX] = 0.0
    cuda.syncthreads()
    if is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z):
        OutputValue = 0.0
        for z in range(0,kernel_size):
            OutputValue += shared_data[(tx)+(ty)*tile_size+(tz+z)*tile_size*tile_size]*d_Filter[z]
        d_dataOut[x_pos, y_pos, z_pos] = OutputValue

@cuda.jit
def cudaConvolve1d_Depth(d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,size_x,size_y,size_z):
    shared_data = cuda.shared.array(shape=0,dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    bx = cuda.blockIdx.x * tile_size
    by = cuda.blockIdx.y * tile_size
    bz = cuda.blockIdx.z * tile_size

    x_pos = bx + tx
    y_pos = by + ty
    z_pos = bz + tz
    tile_id = tz * cache_size * tile_size + ty * tile_size + tx
    if tile_id < tile_size*tile_size:
        tileZ = int(float(tile_id)/tile_size) % tile_size
        tileX = tile_id % tile_size
        input_z = bz + tileZ
        input_x = bx + tileX
        input_y_root = by - kernel_radius

        for stemLength in range(0,cache_size):
            input_y = input_y_root+stemLength
            if is_pixel(input_x,input_y,input_z,size_x,size_y,size_z):
                shared_data[tileZ*tile_size*cache_size + stemLength*tile_size + tileX] = d_dataIn[input_x, input_y, input_z]
            else:
                shared_data[tileZ*tile_size*cache_size + stemLength*tile_size + tileX] = 0.0
    cuda.syncthreads()
    if is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z):
        OutputValue = 0.0
        for y in range(0,kernel_size):
            OutputValue += shared_data[(tx)+(ty+y)*tile_size+(tz)*cache_size*tile_size]*d_Filter[y]
        d_dataOut[x_pos, y_pos, z_pos] = OutputValue"""


##### cuda 3*1d convolution with rotation using the jit package (slow) #####
"""def blur_3_x_1d_simpleborders_CUDA_R(fioIn, iFeature, ImgFilter):

    # Allocate data on device
    stream = cuda.stream()
    dataIn = fioIn.get_fdata().copy()
    dataOut = fioIn.get_fdata().copy()
    d_dataIn = cuda.to_device(dataIn, stream=stream)
    d_dataOut = cuda.to_device(dataOut, stream=stream)
    d_Filter = cuda.to_device(np.array(ImgFilter), stream=stream)

    # Set GPU Property
    kernel_size = len(ImgFilter)
    kernel_radius = int(kernel_size/2)
    tile_size = 10
    threads = (tile_size,tile_size,tile_size)
    blockX = math.ceil(fioIn.shape[0] / tile_size)
    blockY = math.ceil(fioIn.shape[1] / tile_size)
    blockZ = math.ceil(fioIn.shape[2] / tile_size)
    blocks = (blockX,blockY,blockZ)
    cache_size = (tile_size + (kernel_radius*2))
    dimcache = cache_size*tile_size*tile_size*4
    cudaConvolve1d_R_Row[blocks, threads, stream, dimcache](d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,fioIn.shape[0],fioIn.shape[1],fioIn.shape[2])
    stream.synchronize()
    cudaConvolve1d_R_Col[blocks, threads, stream, dimcache](d_dataOut,d_dataIn,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,fioIn.shape[0],fioIn.shape[1],fioIn.shape[2])
    stream.synchronize()
    cudaConvolve1d_R_Depth[blocks, threads, stream, dimcache](d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,fioIn.shape[0],fioIn.shape[1],fioIn.shape[2])
    stream.synchronize()

    fioOut = d_dataOut.copy_to_host(stream=stream)
    stream.synchronize()
    return nib.Nifti1Image(fioOut, fioIn.affine, fioIn.header)

@cuda.jit
def cudaConvolve1d_R_Row(d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,size_x,size_y,size_z):
    shared_data = cuda.shared.array(shape=0,dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    bx = cuda.blockIdx.x * tile_size
    by = cuda.blockIdx.y * tile_size
    bz = cuda.blockIdx.z * tile_size

    x_pos = bx + tx
    y_pos = by + ty
    z_pos = bz + tz
    tile_id = tz * cache_size * tile_size + ty * tile_size + tx
    if tile_id < tile_size*tile_size:
        tileZ = int(float(tile_id)/tile_size) % tile_size
        tileY = tile_id % tile_size
        input_z = bz + tileZ
        input_y = by + tileY
        input_x_root = bx - kernel_radius

        for stemLength in range(0,cache_size):
            input_x = input_x_root+stemLength
            if is_pixel(input_x,input_y,input_z,size_x,size_y,size_z):
                shared_data[tileZ*cache_size*tile_size + tileY*cache_size + stemLength] = d_dataIn[input_x, input_y, input_z]
            else:
                shared_data[tileZ*cache_size*tile_size + tileY*cache_size + stemLength] = 0.0
    cuda.syncthreads()
    if is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z):
        OutputValue = 0.0
        for x in range(0,kernel_size):
            OutputValue += shared_data[(tx+x)+(ty)*cache_size+(tz)*cache_size*tile_size]*d_Filter[x]
        d_dataOut[y_pos, z_pos, x_pos] = OutputValue

@cuda.jit
def cudaConvolve1d_R_Col(d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,size_x,size_y,size_z):
    shared_data = cuda.shared.array(shape=0,dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    bx = cuda.blockIdx.x * tile_size
    by = cuda.blockIdx.y * tile_size
    bz = cuda.blockIdx.z * tile_size

    x_pos = bx + tx
    y_pos = by + ty
    z_pos = bz + tz
    tile_id = tz * tile_size * tile_size + ty * tile_size + tx
    if tile_id < tile_size*tile_size:
        tileY = int(float(tile_id)/tile_size) % tile_size
        tileX = tile_id % tile_size
        input_x = bx + tileX
        input_y = by + tileY
        input_z_root = bz - kernel_radius

        for stemLength in range(0,cache_size):
            input_z = input_z_root+stemLength
            if is_pixel(input_x,input_y,input_z,size_x,size_y,size_z):
                shared_data[stemLength*tile_size*tile_size + tileY*tile_size + tileX] = d_dataIn[input_x, input_y, input_z]
            else:
                shared_data[stemLength*tile_size*tile_size + tileY*tile_size + tileX] = 0.0
    cuda.syncthreads()
    if is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z):
        OutputValue = 0.0
        for z in range(0,kernel_size):
            OutputValue += shared_data[(tx)+(ty)*tile_size+(tz+z)*tile_size*tile_size]*d_Filter[z]
        d_dataOut[x_pos, y_pos, z_pos] = OutputValue

@cuda.jit
def cudaConvolve1d_R_Depth(d_dataIn,d_dataOut,d_Filter,cache_size,tile_size,kernel_size,kernel_radius,size_x,size_y,size_z):
    shared_data = cuda.shared.array(shape=0,dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    bx = cuda.blockIdx.x * tile_size
    by = cuda.blockIdx.y * tile_size
    bz = cuda.blockIdx.z * tile_size

    x_pos = bx + tx
    y_pos = by + ty
    z_pos = bz + tz
    tile_id = tz * cache_size * tile_size + ty * tile_size + tx
    if tile_id < tile_size*tile_size:
        tileZ = int(float(tile_id)/tile_size) % tile_size
        tileX = tile_id % tile_size
        input_z = bz + tileZ
        input_x = bx + tileX
        input_y_root = by - kernel_radius

        for stemLength in range(0,cache_size):
            input_y = input_y_root+stemLength
            if is_pixel(input_x,input_y,input_z,size_x,size_y,size_z):
                shared_data[tileZ*tile_size*cache_size + stemLength*tile_size + tileX] = d_dataIn[input_x, input_y, input_z]
            else:
                shared_data[tileZ*tile_size*cache_size + stemLength*tile_size + tileX] = 0.0
    cuda.syncthreads()
    if is_pixel(x_pos, y_pos, z_pos, size_x, size_y, size_z):
        OutputValue = 0.0
        for y in range(0,kernel_size):
            OutputValue += shared_data[(tx)+(ty+y)*tile_size+(tz)*cache_size*tile_size]*d_Filter[y]
        d_dataOut[x_pos, y_pos, z_pos] = OutputValue"""
