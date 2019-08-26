<snippet>
  <content>
  
# 3D SIFT CUDA

SIFT is an algorithm introduced by David G.Lowe in 1999. 
This code is based on the work of Matthew Towes at École de technologie supérieure ÉTS.
This is a CUDA implémentation in python of the C++ base code. 

## Installation

You will need at least cuda 10 installed on your computer. 
1. Download it !
2. Go in main directory and install the dependencie. 
3. Take care to keep the Flann repository. The pyFlann is in python 2,7. This repository has been updated to python 3.

## Usage

This algorithm is design to extract features from 3D volumes. The main format are accepted ( Nifti and Analyse format)
The algorith will automatically use the first GPU card on your computer. 

    python Sift_main [options] \<input image\> \<output features\>
  
		<input image>: nifti (.nii,.hdr,.nii.gz) or raw input volume (IEEE 32-bit float, little endian).
		<output features>: output file with features.
		[options]
		  -w         : output feature geometry in world coordinates, NIFTI qto_xyz matrix (default is voxel units).
		  -2+        : double input image size.
		  -2-        : halve input image size.

## History

TODO: Need to add fast descriptor computation using BRIEF and 2 other method introduced in the linked publication.

TODO: Need to add best GPU auto selection and option to let user choose GPU card. 

## Credits

Jean-Baptiste CARLUER at École de technologie supérieure ÉTS.

## Publication

TODO: Write publication and link it here

</content>
</snippet>
