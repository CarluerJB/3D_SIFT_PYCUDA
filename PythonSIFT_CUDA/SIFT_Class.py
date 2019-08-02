import numpy as np
import itertools
from functools import *
import math
import operator
import nibabel as nib
import src_common
import operator
import scipy.stats as ss
import time
import copy
import sys


fBlurGradOriHist = 0.5

# Most experiments use 0.5
fHist2ndPeakThreshold = 0.5

def pythag(a,b):
    return math.sqrt(a*a + b*b)

def _svd_max(a,b):
    return a if a>b else b

def _svd_min(a,b):
    return b if a>b else a

def SIGN(a,b):
    return math.fabs(a) if ((b)>=0.0) else -math.fabs(a)

def SingularValueDecomp(mat, w, v, n=3, m=3):
    rv1=np.zeros(n)
    scale=0.0
    anorm=0.0
    g=0.0
    s=0.0
    for i in range(1,n+1):
        l=i+1
        rv1[i-1]=scale*g
        g=0.0
        s=0.0
        scale=0.0
        if i<= m:
            for k in range(i,m+1):
                scale += math.fabs(mat[k-1, i-1])
            if scale:
                for k in range(i,m+1):
                    mat[k-1, i-1] /= scale
                    s += mat[k-1, i-1]*mat[k-1, i-1]
                f=mat[i-1][i-1]
                g=-SIGN(math.sqrt(s), f)
                h=f*g-s
                mat[i-1][i-1]=f-g
                for j in range(l, n+1):
                    for k in range(i, m+1):
                        s=0.0
                        s+=mat[k-1, i-1]*mat[k-i, j-1]
                    f=s/h
                    for k in range(i, m+1):
                        mat[k-1, j-1] += f*mat[k-1, i-1]
                for k in range(i, m+1):
                    mat[k-1, i-1] *= scale
        w[i-1]=scale*g
        g=0.0
        s=0.0
        scale=0.0
        if i<=m and i != n:
            for k in range(l, n+1):
                scale += math.fabs(mat[i-1, k-1])
            if scale:
                for k in range(l, n+1):
                    mat[i-1, k-1] /= scale
                    s += mat[i-1, k-1]*mat[i-1, k-1]
                f=mat[i-1, l-1]
                g = -SIGN(math.sqrt(s), f)
                h= f*g-s
                mat[i-1, l-1]=f-g
                for k in range(l, n+1):
                    rv1[k-1]=mat[i-1, k-1]/h
                for j in range(l, m+1):
                    for k in range(l, n+1):
                        s=0.0
                        s+=mat[j-1, k-1]*mat[i-1, k-1]
                    for k in range(l, n+1):
                        mat[j-1, k-1] += s*rv1[k-1]
                for k in range(l, n+1):
                    mat[i-1, k-1] *= scale
        anorm=_svd_max(anorm, (math.fabs(w[i-1])+math.fabs(rv1[i-1])))
    for i in reversed(range(1, n+1)):
        if i<n:
            if g:
                for j in range(l, n+1):
                    v[j-1, i-1]=(mat[i-1,j-1]/mat[i-1, l-1])/g
                for j in range(l, n+1):
                    for k in range(l, n+1):
                        s=0.0
                        s+= mat[i-1, k-1]*v[k-1, j-1]
                    for k in range(l, n+1):
                        v[k-1, j-1]+=s*v[k-1,i-1]
            for j in range(l, n+1):
                v[i-1, j-1]=0.0
                v[j-1, i-1]=0.0
        v[i-1, i-1]=1.0
        g=rv1[i-1]
        l=i
    for i in reversed(range(1, _svd_min(m,n)+1)):
        l=i+1
        g=w[i-1]
        for j in range(l, n+1):
            mat[i-1, j-1]=0.0
        if(g):
            g=1.0/g
            for j in range(l, n+1):
                for k in range(l, n+1):
                    s=0.0
                    s+=mat[k-1, i-1]*mat[k-1, j-1]
                f=(s/mat[i-1, i-1])*g
                for k in range(i, m+1):
                    mat[k-1, j-1]+=f*mat[k-1, i-1]
            for j in range(i, m+1):
                mat[j-1, i-1]*=g
        else:
            for j in range(i, m+1):
                mat[j-1, i-1]=0.0
            mat[i-1, i-1]+=1
    for k in reversed(range(1, n+1)):
        for its in range(1, 31):
            flag=1
            for l in reversed(range(1, k+1)):
                nm=l-1
                if float((math.fabs(rv1[l-1]+anorm)))==anorm:
                    flag=0
                    break
                if float((math.fabs(w[nm-1]+anorm)))==anorm:
                    break
            if flag:
                c=0.0
                s=1.0
                for i in range(l, k+1):
                    f=s*rv1[i-1]
                    rv1[i-1]=c*rv[i-1]
                    if float(math.fabs(f)+anorm)==anorm:
                        break
                    g=w[i-1]
                    h=pythag(f,g)
                    w[i-1]=h
                    h=1.0/h
                    c=g*h
                    s=-f*h
                    for j in range(1, m+1):
                        y=mat[j-1][nm-1]
                        z=mat[j-1, i-1]
                        mat[j-1, nm-1]=y*c+z*s
                        mat[j-1, i-1]=z*c-y*s
            z=w[k-1]
            if l==k:
                if z<0.0:
                    w[k-1]=-z
                    for j in range(1, n+1):
                        v[j-1, k-1] = -v[j-1, k-1]
                break
            x=w[l-1]
            nm=k-1
            y=w[nm-1]
            g=rv1[nm-1]
            h=rv1[k-1]
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
            g=pythag(f, 1.0)
            f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x
            c=1.0
            s=1.0
            for j in range(l, nm+1):
                i=j+1
                g=rv1[i-1]
                y=w[i-1]
                h=s*g
                g=c*g
                z=pythag(f,h)
                rv1[j-1]=z
                c=f/z
                s=h/z
                f=x*c+g*s
                g = g*c-x*s
                h=y*s
                y *= c
                for jj in range(1, n+1):
                    x=v[jj-1, j-1]
                    z=v[jj-1, i-1]
                    v[jj-1, j-1]=x*c+z*s
                    v[jj-1, j-1]=z*c-x*s
                z=pythag(f,h)
                w[j-1]=z
                if z:
                    z=1.0/z
                    c=f*z
                    s=h*z
                f=c*g+s*y;
                x=c*y-s*g;
                for jj in range(1,m+1):
                    y=mat[jj-1, j-1]
                    z=mat[jj-1, i-1]
                    mat[jj-1, j-1]=y*c+z*s
                    mat[jj-1, i-1]=z*c-y*s
            rv1[l-1]=0.0;
            rv1[k-1]=f;
            w[k-1]=x;
    return w, v

def prod(factors):
    return reduce(operator.mul, factors, 1)

# TOOLBOX
def mult_3x3_matrix(mat1, mat2):
    mat_out = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for ii in range(3):
                mat_out[i][j] += mat1[i][ii] * mat2[ii][j]
    return mat_out

def mult_3x3(mat1, list1):

    #NO LOOP Method
    """
    list1=np.full((3,3), list1).T
    mat1=mat1.T
    list_out=sum(list1*mat1)
    end = time.time()
    print(end-start)"""

    #LOOP Method (Faster for this kind of little operation)
    list_out = np.zeros(3)
    for j, i in list(itertools.product(range(3), repeat=2)):
        list_out[i] += mat1[i,j]*list1[j]

    return list_out

def mult_4x4_vector(vec_in, mat):
    # OPTIMIZE: no concrete loop use inline
    out=np.zeros((4))
    for i in range(4):
        for j in range(4):
            out[i]+=mat[i*4+j]*vec_in[j]
    return out[0],out[1],out[2]

def invert_3x3(matIn):
    # Init version (c++ like), slow on python
    """a11 = float(matIn[0,0])
    a21 = float(matIn[1,0])
    a31 = float(matIn[2,0])
    a12 = float(matIn[0,1])
    a22 = float(matIn[1,1])
    a32 = float(matIn[2,1])
    a13 = float(matIn[0,2])
    a23 = float(matIn[1,2])
    a33 = float(matIn[2,2])
    det = float(a11*(a33*a22 - a32*a23) - a21*(a33*a12 - a32*a13) + a31*(a23*a12 - a22*a13))
    div = 1/det
    matOut=np.zeros((3,3))
    matOut[0,0] = (a33*a22 - a32*a23)*div
    matOut[1,0] = -(a33*a21 - a31*a23)*div
    matOut[2,0] = (a32*a21 - a31*a22)*div
    matOut[0,1] = -(a33*a12 - a32*a13)*div
    matOut[1,1] = (a33*a11 - a31*a13)*div
    matOut[2,1] = -(a32*a11 - a31*a12)*div
    matOut[0,2] = (a23*a12 - a22*a13)*div
    matOut[1,2] = -(a23*a11 - a21*a13)*div
    matOut[2,2] = (a22*a11 - a21*a12)*div"""
    return matIn.transpose()

def _fioDetermineInterpCoord(fx, fMinX, fMaxX):
    if fx<fMinX+0.5:
        return int(fMinX), 1.0
    elif(fx >= fMaxX-0.5):
        return int(fMaxX - 2), 0.0
    else:
        fxMinuxHalf = float(fx-0.5)
        iXCoord = int(math.floor(fxMinuxHalf))
        return iXCoord, 1.0-(fxMinuxHalf-float(iXCoord))

# THIS IS OLD FUNCTION (SLOW)
def fioGenerateEdgeImage3D(fioImg, fioDx, fioDy, fioDz):
    if fioImg.shape[0]*fioImg.shape[1]*fioImg.shape[2] != fioDx.shape[0]*fioDx.shape[1]*fioDx.shape[2] or fioImg.shape[0]*fioImg.shape[1]*fioImg.shape[2] != fioDy.shape[0]*fioDy.shape[1]*fioDy.shape[2] or fioDy.shape[0]*fioDy.shape[1]*fioDy.shape[2] != fioDx.shape[0]*fioDx.shape[1]*fioDx.shape[2] :
        return 0
    fioDx[1:fioImg.shape[0]-1,1:fioImg.shape[1]-1,1:fioImg.shape[2]-1] = fioImg[2:fioImg.shape[0], 1:fioImg.shape[1]-1, 1:fioImg.shape[2]-1] - fioImg[0:fioImg.shape[0]-2, 1:fioImg.shape[1]-1, 1:fioImg.shape[2]-1]
    fioDy[1:fioImg.shape[0]-1,1:fioImg.shape[1]-1,1:fioImg.shape[2]-1] = fioImg[1:fioImg.shape[0]-1, 2:fioImg.shape[1], 1:fioImg.shape[2]-1] - fioImg[1:fioImg.shape[0]-1, 0:fioImg.shape[1]-2, 1:fioImg.shape[2]-1]
    fioDz[1:fioImg.shape[0]-1,1:fioImg.shape[1]-1,1:fioImg.shape[2]-1] = fioImg[1:fioImg.shape[0]-1, 1:fioImg.shape[1]-1, 2:fioImg.shape[2]] - fioImg[1:fioImg.shape[0]-1, 1:fioImg.shape[1]-1, 0:fioImg.shape[2]-2]

    for z,y,x in list(itertools.product(range(1,fioImg.shape[0]-1), range(1,fioImg.shape[1]-1), range(1,fioImg.shape[2]-1))):
        fioDx[x, y, z] = fioImg[x+1,y,z]-fioImg[x-1,y,z]
        fioDy[x, y, z] = fioImg[x,y+1,z]-fioImg[x,y-1,z]
        fioDz[x, y, z] = fioImg[x,y,z+1]-fioImg[x,y,z-1]
    return fioDx, fioDy, fioDz

def fioGetPixelTrilinearInterp(fio, x, y, z):
    iX, fXCont = _fioDetermineInterpCoord(x, 0, fio.shape[0])
    iY, fYCont = _fioDetermineInterpCoord(y, 0, fio.shape[1])
    iZ, fZCont = _fioDetermineInterpCoord(z, 0, fio.shape[2])
    f000 = fio[iX+0, iY+0, iZ+0]
    f100 = fio[iX+1, iY+0, iZ+0]
    f010 = fio[iX+0, iY+1, iZ+0]
    f110 = fio[iX+1, iY+1, iZ+0]
    f001 = fio[iX+0, iY+0, iZ+1]
    f101 = fio[iX+1, iY+0, iZ+1]
    f011 = fio[iX+0, iY+1, iZ+1]
    f111 = fio[iX+1, iY+1, iZ+1]

    fn00 = fXCont * f000 + (1.0-fXCont)*f100
    fn01 = fXCont * f001 + (1.0-fXCont)*f101
    fn10 = fXCont * f010 + (1.0-fXCont)*f110
    fn11 = fXCont * f011 + (1.0-fXCont)*f111

    fnn0 = fYCont*fn00 + (1.0 - fYCont)*fn10
    fnn1 = fYCont*fn01 + (1.0 - fYCont)*fn11

    fnnn = fZCont*fnn0 + (1.0 - fZCont)*fnn1
    return fnnn

def fioGetPixelTrilinearInterp_flat(fio, x, y, z):
    iX, fXCont = _fioDetermineInterpCoord(x, 0, fio.x)
    iY, fYCont = _fioDetermineInterpCoord(y, 0, fio.y)
    iZ, fZCont = _fioDetermineInterpCoord(z, 0, fio.z)
    f000 = fio._cpu_data[iX+0 + (iY+0)*fio.x + (iZ+0)*fio.x*fio.y]
    f100 = fio._cpu_data[iX+1 + (iY+0)*fio.x + (iZ+0)*fio.x*fio.y]
    f010 = fio._cpu_data[iX+0 + (iY+1)*fio.x + (iZ+0)*fio.x*fio.y]
    f110 = fio._cpu_data[iX+1 + (iY+1)*fio.x + (iZ+0)*fio.x*fio.y]
    f001 = fio._cpu_data[iX+0 + (iY+0)*fio.x + (iZ+1)*fio.x*fio.y]
    f101 = fio._cpu_data[iX+1 + (iY+0)*fio.x + (iZ+1)*fio.x*fio.y]
    f011 = fio._cpu_data[iX+0 + (iY+1)*fio.x + (iZ+1)*fio.x*fio.y]
    f111 = fio._cpu_data[iX+1 + (iY+1)*fio.x + (iZ+1)*fio.x*fio.y]

    fn00 = fXCont * f000 + (1.0-fXCont)*f100
    fn01 = fXCont * f001 + (1.0-fXCont)*f101
    fn10 = fXCont * f010 + (1.0-fXCont)*f110
    fn11 = fXCont * f011 + (1.0-fXCont)*f111

    fnn0 = fYCont*fn00 + (1.0 - fYCont)*fn10
    fnn1 = fYCont*fn01 + (1.0 - fYCont)*fn11

    fnnn = fZCont*fnn0 + (1.0 - fZCont)*fnn1
    return fnnn

def fioIncPixelTrilinearInterp(fio, xyz, fValue, iFeature=1):
    iX, fXCont = _fioDetermineInterpCoord(xyz[0], 0, fio.shape[0])
    iY, fYCont = _fioDetermineInterpCoord(xyz[1], 0, fio.shape[1])
    iZ, fZCont = _fioDetermineInterpCoord(xyz[2], 0, fio.shape[2])

    fio[iX+0*iFeature, iY+0*iFeature, iZ+0*iFeature] += fValue*fXCont*fYCont*fZCont
    fio[iX+1*iFeature, iY+0*iFeature, iZ+0*iFeature] += fValue*(1.0-fXCont)*fYCont*fZCont
    fio[iX+0*iFeature, iY+1*iFeature, iZ+0*iFeature] += fValue*fXCont*(1.0-fYCont)*fZCont
    fio[iX+1*iFeature, iY+1*iFeature, iZ+0*iFeature] += fValue*(1.0-fXCont)*(1.0-fYCont)*fZCont
    fio[iX+0*iFeature, iY+0*iFeature, iZ+1*iFeature] += fValue*fXCont*fYCont*(1.0-fZCont)
    fio[iX+1*iFeature, iY+0*iFeature, iZ+1*iFeature] += fValue*(1.0-fXCont)*fYCont*(1.0-fZCont)
    fio[iX+0*iFeature, iY+1*iFeature, iZ+1*iFeature] += fValue*fXCont*(1.0-fYCont)*(1.0-fZCont)
    fio[iX+1*iFeature, iY+1*iFeature, iZ+1*iFeature] += fValue*(1.0-fXCont)*(1.0-fYCont)*(1.0-fZCont)
    return fio

def fioIncPixelTrilinearInterp2(fio, xyz, fValue, shape0, shape1, shape2, maxfeature, iFeature):

    iX, fXCont = _fioDetermineInterpCoord(float(xyz[0]), float(0), float(shape0))
    iY, fYCont = _fioDetermineInterpCoord(float(xyz[1]), float(0), float(shape1))
    iZ, fZCont = _fioDetermineInterpCoord(float(xyz[2]), float(0), float(shape2))

    fio[((iZ+0)*shape1*shape0 +(iY+0)*shape0+(iX+0))*maxfeature+iFeature] += fValue*fXCont*fYCont*fZCont
    fio[((iZ+0)*shape1*shape0 +(iY+0)*shape0+(iX+1))*maxfeature+iFeature] += fValue*(1.0-fXCont)*fYCont*fZCont
    fio[((iZ+0)*shape1*shape0 +(iY+1)*shape0+(iX+0))*maxfeature+iFeature] += fValue*fXCont*(1.0-fYCont)*fZCont
    fio[((iZ+0)*shape1*shape0 +(iY+1)*shape0+(iX+1))*maxfeature+iFeature] += fValue*(1.0-fXCont)*(1.0-fYCont)*fZCont
    fio[((iZ+1)*shape1*shape0 +(iY+0)*shape0+(iX+0))*maxfeature+iFeature] += fValue*fXCont*fYCont*(1.0-fZCont)
    fio[((iZ+1)*shape1*shape0 +(iY+0)*shape0+(iX+1))*maxfeature+iFeature] += fValue*(1.0-fXCont)*fYCont*(1.0-fZCont)
    fio[((iZ+1)*shape1*shape0 +(iY+1)*shape0+(iX+0))*maxfeature+iFeature] += fValue*fXCont*(1.0-fYCont)*(1.0-fZCont)
    fio[((iZ+1)*shape1*shape0 +(iY+1)*shape0+(iX+1))*maxfeature+iFeature] += fValue*(1.0-fXCont)*(1.0-fYCont)*(1.0-fZCont)
    return fio

def fioIncPixelTrilinearInterpFlat(fio, xyz, fValue, iFeature=1):
    iX, fXCont = _fioDetermineInterpCoord(xyz[0], 0, fio.x)
    iY, fYCont = _fioDetermineInterpCoord(xyz[1], 0, fio.y)
    iZ, fZCont = _fioDetermineInterpCoord(xyz[2], 0, fio.z)

    fio._cpu_data[(iX+0)*iFeature+ (iY+0)*iFeature*fio.x + (iZ+0)*iFeature*fio.x*fio.y] += fValue*fXCont*fYCont*fZCont
    fio._cpu_data[(iX+1)*iFeature+ (iY+0)*iFeature*fio.x + (iZ+0)*iFeature*fio.x*fio.y] += fValue*(1.0-fXCont)*fYCont*fZCont
    fio._cpu_data[(iX+0)*iFeature+ (iY+1)*iFeature*fio.x + (iZ+0)*iFeature*fio.x*fio.y] += fValue*fXCont*(1.0-fYCont)*fZCont
    fio._cpu_data[(iX+1)*iFeature+ (iY+1)*iFeature*fio.x + (iZ+0)*iFeature*fio.x*fio.y] += fValue*(1.0-fXCont)*(1.0-fYCont)*fZCont
    fio._cpu_data[(iX+0)*iFeature+ (iY+0)*iFeature*fio.x + (iZ+1)*iFeature*fio.x*fio.y] += fValue*fXCont*fYCont*(1.0-fZCont)
    fio._cpu_data[(iX+1)*iFeature+ (iY+0)*iFeature*fio.x + (iZ+1)*iFeature*fio.x*fio.y] += fValue*(1.0-fXCont)*fYCont*(1.0-fZCont)
    fio._cpu_data[(iX+0)*iFeature+ (iY+1)*iFeature*fio.x + (iZ+1)*iFeature*fio.x*fio.y] += fValue*fXCont*(1.0-fYCont)*(1.0-fZCont)
    fio._cpu_data[(iX+1)*iFeature+ (iY+1)*iFeature*fio.x + (iZ+1)*iFeature*fio.x*fio.y] += fValue*(1.0-fXCont)*(1.0-fYCont)*(1.0-fZCont)
    return fio

def finddet(a1, a2, a3, b1, b2, b3, c1, c2, c3):
    return (a1*b2*c3) - (a1*b3*c2) - (a2*b1*c3) + (a3*b1*c2) + (a2*b3*c1) - (a3*b2*c1)

def interpolate_extremum_quadratic(x0, x1, x2, fx0, fx1, fx2):
    if not((fx1 < fx0 and fx1 < fx2) or (fx1 > fx0 and fx1 > fx2)):
        print(str(fx0) + "\t" + str(fx1) + "\t" + str(fx2) + "\n")
        assert(0)
        return x1
    a1 = pow(x0,2)
    b1 = x0
    c1 = 1
    d1 = fx0
    a2 = pow(x1,2)
    b2 = x1
    c2 = 1
    d2 = fx1
    a3 = pow(x2,2)
    b3 = x2
    c3 = 1
    d3 = fx2

    det = finddet(a1, a2, a3, b1, b2, b3, c1, c2, c3)
    detx = finddet(d1, d2, d3, b1, b2, b3, c1, c2, c3)
    dety = finddet(a1, a2, a3, d1, d2, d3, c1, c2, c3)
    detz = finddet(a1, a2, a3, b1, b2, b3, d1, d2, d3)

    if sum((d1, d2, d3)) == 0 :
        if det==0:
            print("\n Infinite Solutions\n ")
        elif det != 0:
            print("\n x=0\n y=0, \n z=0\n ")
    elif det != 0:
        if detx != 0:
            return dety / (-2.0*detx)
    elif (det == 0 and detx == 0 and dety == 0 and detz == 0):
        print("\n Infinite Solutions\n ")
    else:
        print("No Solution\n ")
    return x1

def fioGenerateEdgeImage3D(fioImg):
    fioDx=np.zeros(fioImg.shape)
    fioDy=np.zeros(fioImg.shape)
    fioDz=np.zeros(fioImg.shape)
    fioDz[1:fioImg.shape[0]-1,1:fioImg.shape[1]-1,1:fioImg.shape[2]-1]=fioImg[2:fioImg.shape[0],1:fioImg.shape[1]-1,1:fioImg.shape[2]-1]-fioImg[0:fioImg.shape[0]-2,1:fioImg.shape[1]-1,1:fioImg.shape[2]-1]
    fioDy[1:fioImg.shape[0]-1,1:fioImg.shape[1]-1,1:fioImg.shape[2]-1]=fioImg[1:fioImg.shape[0]-1,2:fioImg.shape[1],1:fioImg.shape[2]-1]-fioImg[1:fioImg.shape[0]-1,0:fioImg.shape[1]-2,1:fioImg.shape[2]-1]
    fioDx[1:fioImg.shape[0]-1,1:fioImg.shape[1]-1,1:fioImg.shape[2]-1]=fioImg[1:fioImg.shape[0]-1,1:fioImg.shape[1]-1,2:fioImg.shape[2]]-fioImg[1:fioImg.shape[0]-1,1:fioImg.shape[1]-1,0:fioImg.shape[2]-2]
    return fioDx, fioDy, fioDz

# ClassBox

class FEATUREIO:
    def copy(self, fio):
        self.x=fio.x
        self.y=fio.y
        self.z=fio.z

class FEATUREDATA:
    def copy(self, fio):
        self.x=fio.x
        self.y=fio.y
        self.z=fio.z

class Location_Value_XYZ:

    def __init__(self, x, y, z,fValue, fValueH=None, fValueL=None):
        self.x=x
        self.y=y
        self.z=z
        self.fValue=fValue
        self.fValueH=fValueH
        self.fValueL=fValueL

class Extremum(Location_Value_XYZ):
    def __init__(self, pfioH, pfioL, bDense):
        self.pfioH = pfioH
        self.pfioL = pfioL
        self.bDense = bDense
    def add_ext(self, iNeighbourCount, x, y, z, fValue, piNeighbourIndices):
        self.lva=Location_Value_XYZ(x, y, z, fValue)
        self.iNeighbourCount=iNeighbourCount
        self.piNeighbourIndices=piNeighbourIndices


class Feature3DInfo:
    FEATURE_3D_PCS = 64
    PC_ARRAY_SIZE = 64
    INFO_FLAG_MIN0MAX1 = 0x00000010
    INFO_FLAG_REORIENT = 0x00000020
    INFO_FLAG_LINE = 0x00000100
    def __init__(self):
        self.x=0.0
        self.y=0.0
        self.z=0.0
        self.scale=0.0
        self.ori=np.eye(3,3)
        self._fOriInv=np.zeros((3, 3))
        self.eigs=np.zeros(3)
        self.m_pfPC=np.zeros(self.FEATURE_3D_PCS)
        self.m_uiInfo=0

    def SimilarityTransform(self, Mat4x4):
        fP0 = [self.x,self.y,self.z,1]
        self.x, self.y, self.z = mult_4x4_vector(fP0,Mat4x4)
        fScaleSum = 0
        fRot3x3=np.zeros((3, 3))
        for i in range(3):
            fSumsqr = pow((Mat4x4[4*i]+0), 2) + pow((Mat4x4[4*i+1]), 2) + pow((Mat4x4[4*i+2]), 2)
            fScaleSum = fScaleSum+fSumsqr if fSumsqr > 0 else 0
            fRot3x3[i] = Mat4x4[4*i]
            fSumSqr = pow((Mat4x4[4*i]),2) + pow((Mat4x4[4*i+1]),2) + pow((Mat4x4[4*i+2]),2)
            fDiv = 1.0 / np.sqrt(fSumSqr)
            fRot3x3[i] = fRot3x3[i]*fDiv if fSumSqr>0 else (1,0,0)
        fScaleSum/=3
        self.scale *= fScaleSum
        self.ori=(mult_3x3_matrix(((self.ori).transpose()),fRot3x3)).transpose()

    def add_coord_bInterpolate(self, x, y, z, scale):
        self.x = x
        self.y = y
        self.z = z
        self.scale = 2*scale

    def conv_loc_2_subpixel_prec(self, value):
        self.x += value
        self.y += value
        self.z += value

    def NormalizeDataRankedPCs(self):
        self.m_pfPC[:]=ss.rankdata(self.m_pfPC[:], method='ordinal')
        self.m_pfPC[:]-=1

    def interpolate_discrete_3D_point_flat(self, fioC, ix, iy, iz):
        self.x = interpolate_extremum_quadratic(ix-1, ix, ix+1, fioC._cpu_data[(ix-1)+ iy*fioC.x+ iz*fioC.x*fioC.y], fioC._cpu_data[ix + iy*fioC.x + iz*fioC.x*fioC.y], fioC._cpu_data[(ix+1) + iy*fioC.x + iz*fioC.x*fioC.y])
        self.y = interpolate_extremum_quadratic(iy-1, iy, iy+1, fioC._cpu_data[ix+ (iy-1)*fioC.x+ iz*fioC.x*fioC.y], fioC._cpu_data[ix + iy*fioC.x + iz*fioC.x*fioC.y], fioC._cpu_data[ix + (iy+1)*fioC.x + iz*fioC.x*fioC.y])
        self.z = interpolate_extremum_quadratic(iz-1, iz, iz+1, fioC._cpu_data[ix+ iy*fioC.x+ (iz-1)*fioC.x*fioC.y], fioC._cpu_data[ix + iy*fioC.x + iz*fioC.x*fioC.y], fioC._cpu_data[ix + iy*fioC.x + (iz+1)*fioC.x*fioC.y])

    def interpolate_discrete_3D_point(self, fioC, ix, iy, iz):
        self.x = interpolate_extremum_quadratic(ix-1, ix, ix+1, fioC[ix-1, iy, iz], fioC[ix, iy, iz], fioC[ix+1, iy, iz])
        self.y = interpolate_extremum_quadratic(iy-1, iy, iy+1, fioC[ix, iy-1, iz], fioC[ix, iy, iz], fioC[ix, iy+1, iz])
        self.z = interpolate_extremum_quadratic(iz-1, iz, iz+1, fioC[ix, iy, iz-1], fioC[ix, iy, iz], fioC[ix, iy, iz+1])

    def interpolate_scale(self, ScaleH, ScaleC, ScaleL, fScaleH, fScaleC, fScaleL):
        self.scale = 2*interpolate_extremum_quadratic(ScaleH, ScaleC, ScaleL, fScaleH, fScaleC, fScaleL)

    def SortEigenDecomp(self):
        for i in range(3):
            for j in range(i+1, 3):
                if self.eigs[i]<self.eigs[j]:
                    t = self.eigs[j]
                    self.eigs[j] = self.eigs[i]
                    self.eigs[i]=t
                    for k in range(3):
                        t=self.ori[k,j]
                        self.ori[k,j]=self.ori[k,i]
                        self.ori[k,i]=t


class Feature3DData:
    def __init__(self):
        self.FEATURE_3D_DIM = 11
        self.data_zyx=np.zeros((self.FEATURE_3D_DIM,self.FEATURE_3D_DIM,self.FEATURE_3D_DIM))
        self.fioSample=np.zeros((self.FEATURE_3D_DIM,self.FEATURE_3D_DIM,self.FEATURE_3D_DIM))

class Feature3D(Feature3DInfo, Feature3DData):

    def __init__(self):
        Feature3DInfo.__init__(self)
        Feature3DData.__init__(self)

    def NormalizeData(self):
        fMin = np.min(self.data_zyx)
        fMax = np.max(self.data_zyx)
        fSum = np.sum(self.data_zyx)
        fMean = fSum/pow(self.FEATURE_3D_DIM, 3)
        self.data_zyx -= fMean

        fSumSqr = np.sum(self.data_zyx* self.data_zyx)
        fDiv = 1.0 / float(np.sqrt(fSumSqr))
        self.data_zyx *= fDiv
        fSumSqr = np.sum(self.data_zyx* self.data_zyx)
        return fMean, fSumSqr, fMin, fMax

    def sampleImage3D(self, fioImg):
        fImageRad = float(2.0*self.scale)
        iRadMax = int(fImageRad + 2)
        if self.x - iRadMax < 0 or self.y - iRadMax < 0 or self.z - iRadMax < 0 or self.x + iRadMax >= fioImg.x or self.y + iRadMax >= fioImg.y or self.z + iRadMax >= fioImg.z:
            return 1
        #print("\nSample : ", self.ori)
        fOriInv = invert_3x3(self.ori)
        iSampleRad = int(self.FEATURE_3D_DIM/2)
        fScale = fImageRad/float(iSampleRad)

        for z, y, x in list(itertools.product(range(-iSampleRad, iSampleRad+1), repeat=3)):
            # 1) rotate feature coordinate to image coordinate
            xyz_img = mult_3x3(fOriInv, [x,y,z])
            # 2) scale feature magnitude
            xyz_img*=fScale
            # 3) Translated to current feature center
            xyz_img[0]+=self.x
            xyz_img[1]+=self.y
            xyz_img[2]+=self.z
            # 4) Interpolate_pixel
            fPixel= 0 if (xyz_img[0] < 0 or xyz_img[0] >= fioImg.x or
						xyz_img[0] < 0 or xyz_img[0] >= fioImg.x or
						xyz_img[0] < 0 or xyz_img[0] >= fioImg.x)\
                        else fioGetPixelTrilinearInterp_flat(fioImg, xyz_img[0],xyz_img[1], xyz_img[2])
            """if (xyz_img[0] < 0 or xyz_img[0] >= fioImg.x or
						xyz_img[0] < 0 or xyz_img[0] >= fioImg.x or
						xyz_img[0] < 0 or xyz_img[0] >= fioImg.x):
                fPixel = 0
            else:
                fPixel = fioGetPixelTrilinearInterp_flat(fioImg, xyz_img[0],xyz_img[1], xyz_img[2])"""
            # 5) Set pixel into sample volume
            self.fioSample[x+iSampleRad,y+iSampleRad, z+iSampleRad] = fPixel
        return 0

    # Determine orientation component of feat3D
    def determineOrientation3D(self):
        fioImg = self.data_zyx
        fioDx, fioDy, fioDz = fioGenerateEdgeImage3D(fioImg)
        fMat = np.zeros((3, 3))
        fRadius = int(fioImg.shape[0]/2)
        fRadiusSqr = fRadius*fRadius
        iSampleCount = 0
        pfEdge = np.zeros(3)

        for zz,yy,xx in list(itertools.product(range(fioImg.shape[0]), range(fioImg.shape[1]), range(fioImg.shape[2]))):
            dz = float(zz - int(fioImg.shape[2]/2))
            dy = float(yy - int(fioImg.shape[1]/2))
            dx = float(xx - int(fioImg.shape[0]/2))
            if (dz*dz + dy*dy + dx*dx) < fRadiusSqr:
                # Keep this Sample
                pfEdge[0] = fioDx[xx, yy, zz]
                pfEdge[1] = fioDy[xx, yy, zz]
                pfEdge[2] = fioDz[xx, yy, zz]
                for j, i in list(itertools.product(range(3), repeat=2)):
                    fMat[i,j] += pfEdge[i]*pfEdge[j]

        self.ori[:,:], self.eigs[:], t = np.linalg.svd(fMat)
        #self.eigs, self.ori = SingularValueDecomp(fMat, self.eigs, self.ori)
        self.SortEigenDecomp()
        return 0

    def generate_Feature3D(self, fioImg, fEigThres, vecFeat, bReorientedFeatures=1):
        if self.sampleImage3D(fioImg ) != 0:
            return vecFeat
        fMinPixel = np.min(self.fioSample)
        fMaxPixel = np.max(self.fioSample)
        self.data_zyx=np.transpose(self.fioSample, (2,1,0))
        self.NormalizeData()
        if self.determineOrientation3D():
            return vecFeat
        fEigSum = float(sum(self.eigs))
        fEigPrd = float(prod(self.eigs))
        fEigSumProd = pow(fEigSum, 3)
        fRatio = float((27.0*fEigPrd)/fEigSumProd)
        fMaxRatio = -1 if fRatio < -1 else fRatio
        fMinRatio = 2  if fRatio > 2 else fRatio
        if (fEigSumProd<fEigThres*fEigPrd or fEigThres<0):
            pass
        else:
            return vecFeat
        self.m_uiInfo &= ~self.INFO_FLAG_REORIENT
        Feat=copy.deepcopy(self)
        vecFeat.append(Feat)

        if not bReorientedFeatures:
            return vecFeat
        iOrientationsFound=0
        # TODO: INCLUDE_EIGENORIENTATION_FEATURE func
        iOrientationsFound, OriMatrix = self.determineCanonicalOrientation3D(30)
        for iOri in range(iOrientationsFound):
            canonicalFeat = copy.deepcopy(self)

            for j, i  in list(itertools.product(range(3), repeat=2)):
                canonicalFeat.ori[i,j] = OriMatrix[iOri][i*3+j]
            #print("CANONIC", canonicalFeat.ori)
            if canonicalFeat.sampleImage3D(fioImg) != 0:
                continue
            canonicalFeat.data_zyx=np.transpose(canonicalFeat.fioSample, (2,1,0))
            canonicalFeat.m_uiInfo |= canonicalFeat.INFO_FLAG_REORIENT
            vecFeat.append(canonicalFeat)
        return vecFeat

    def determineCanonicalOrientation3D(self, iMaxOri):
        #if (self.x < 119.4122 and self.x > 119.4120):
        np.set_printoptions(threshold=sys.maxsize, precision=5)
        lvaPeaks=[]
        OriData=[]
        fioDx, fioDy, fioDz = fioGenerateEdgeImage3D(self.data_zyx)
        fRadius = float(int((self.data_zyx).shape[0]/2.0))
        fRadiusSqr = fRadius*fRadius
        iSampleCount=0
        pfOriMatrix=[]
        es = Extremum(False, False, 0)
        fT0 = Feature3D()
        fT1 = Feature3D()
        fT2 = Feature3D()
        fT0.data_zyx=np.zeros((self.FEATURE_3D_DIM,self.FEATURE_3D_DIM,self.FEATURE_3D_DIM))

        for x, y, z in list(itertools.product(range((self.data_zyx).shape[0]), repeat=3)):
            dz=float(z-fRadius)
            dy=float(y-fRadius)
            dx=float(x-fRadius)
            if dz*dz + dy*dy + dx*dx < fRadiusSqr:
                pfEdge=np.zeros(3)
                pfEdge[0]=fioDx[x,y,z]
                pfEdge[1]=fioDy[x,y,z]
                pfEdge[2]=fioDz[x,y,z]
                fEdgeMagSqr = pfEdge[0]*pfEdge[0] + pfEdge[1]*pfEdge[1] + pfEdge[2]*pfEdge[2]
                if fEdgeMagSqr==0:
                    continue
                fEdgeMag = np.sqrt(fEdgeMagSqr)
                iSampleCount+=1
                pfEdgeUnit=(pfEdge*fRadius)/fEdgeMag
                pfEdgeUnit+=fRadius
                fT0.data_zyx = fioIncPixelTrilinearInterp(fT0.data_zyx, pfEdgeUnit+0.5, fEdgeMag)

        iSampleCount=len(np.where(fT0.data_zyx>0)[0])


        fT2.data_zyx, iReturn=src_common.gb3d_blur3d(fT0.data_zyx, fT2.data_zyx, fSigma=fBlurGradOriHist, fMinValue=0.01, witchCuda=0, fioTemp=None)

        lvaPeaks=src_common.detect_local_extremumMax(fT2.data_zyx, es, lvaPeaks)
        lvaPeaks.sort(key=operator.attrgetter('fValue'), reverse = True)



        iMin = min((fT0.data_zyx).shape[2], iMaxOri, len(lvaPeaks))

        FeatTab=np.zeros(3)
        for i in range(iMin):
            QuickFeat = Feature3D()
            QuickFeat.interpolate_discrete_3D_point(fT2.data_zyx,lvaPeaks[i].x, lvaPeaks[i].y, lvaPeaks[i].z)
            QuickFeat.conv_loc_2_subpixel_prec(-fRadius)
            FeatTab[0]=QuickFeat.x
            FeatTab[1]=QuickFeat.y
            FeatTab[2]=QuickFeat.z
            FeatTab=src_common.vec3D_norm_3d(FeatTab)
            QuickFeat.x=FeatTab[0]
            QuickFeat.y=FeatTab[1]
            QuickFeat.z=FeatTab[2]
            OriData.append(QuickFeat)

        #print(self.x, self.y, self.z)

        # WORK UNTIL HERE
        iOrientationsReturned = 0
        self.data_zyx=np.zeros((self.FEATURE_3D_DIM,self.FEATURE_3D_DIM,self.FEATURE_3D_DIM))
        iMin = min((self.data_zyx).shape[2], len(lvaPeaks))
        for i in range(iMin):
            pfP1=np.zeros(3)
            pfP1[0]=OriData[i].x
            pfP1[1]=OriData[i].y
            pfP1[2]=OriData[i].z
            if iOrientationsReturned==iMaxOri:
                break
            if(lvaPeaks[i].fValue<0.8*lvaPeaks[0].fValue):
                break
            fT0.data_zyx=np.zeros((self.FEATURE_3D_DIM,self.FEATURE_3D_DIM,self.FEATURE_3D_DIM))
            for z, y, x in list(itertools.product(range(self.data_zyx.shape[0]), repeat=3)):
                dz=float(z-(fRadius))
                dy=float(y-(fRadius))
                dx=float(x-(fRadius))

                if dz*dz + dy*dy + dx*dx < fRadiusSqr:
                    pfEdge=np.zeros(3)
                    pfEdge[0]=fioDx[z,y,x]
                    pfEdge[1]=fioDy[z,y,x]
                    pfEdge[2]=fioDz[z,y,x]
                    fEdgeMag = src_common.vec3D_mag(pfEdge)
                    if fEdgeMag==0:
                        continue
                    fEdgeUnit=src_common.vec3D_norm_3d(pfEdge)
                    # Remove component parallel to primary orientation
                    #print('{:f}'.format(fEdgeMag))
                    fParallelMag = src_common.vec3D_dot_3d(pfP1, fEdgeUnit)
                    pVecPerp=np.zeros(3)
                    pVecPerp[0]=fEdgeUnit[0]-(fParallelMag*OriData[i].x)
                    pVecPerp[1]=fEdgeUnit[1]-(fParallelMag*OriData[i].y)
                    pVecPerp[2]=fEdgeUnit[2]-(fParallelMag*OriData[i].z)
                    pVecPerp=src_common.vec3D_norm_3d(pVecPerp)
                    pVecPerp*=fRadius
                    pVecPerp+=fRadius
                    pVecPerp=pVecPerp
                    fT0.data_zyx = fioIncPixelTrilinearInterp(fT0.data_zyx, pVecPerp+0.5, fEdgeMag)

            fioT2=np.zeros(fT0.data_zyx.shape)
            fioT2, iReturn=src_common.gb3d_blur3d(fT0.data_zyx, fioT2, fSigma=fBlurGradOriHist, fMinValue=0.01, witchCuda=0, fioTemp=None)
            lvaPeaks2=[]
            """for z in range(self.FEATURE_3D_DIM):
                for y in range(self.FEATURE_3D_DIM):
                    for x in range(self.FEATURE_3D_DIM):
                        print('{:f}'.format(fioT2[x, y, z],6), " ", end='')
                    print("")
                print("\n")"""
            #src_common.transform_showCPU(fioT2)
            lvaPeaks2=src_common.detect_local_extremumMax(fioT2, es, lvaPeaks2)
            lvaPeaks2.sort(key=operator.attrgetter('fValue'), reverse = True)
            #print("\t", OriData[i].x, OriData[i].y, OriData[i].z)

            """for peak in lvaPeaks2:
                print("\t\t", peak.x, peak.y, peak.z, peak.fValue)"""
            OriData2=[]
            pfP2=np.zeros(3)
            #if (self.x < 119.4122 and self.x > 119.4120):
            for j in range(len(lvaPeaks2)):
                if iOrientationsReturned>=(self.data_zyx).shape[2]:
                    break
                if iOrientationsReturned >= iMaxOri:
                    break
                if lvaPeaks2[j].fValue < fHist2ndPeakThreshold*lvaPeaks[0].fValue:
                    break
                QuickFeat2 = Feature3D()
                pfP2[0]=lvaPeaks2[j].x -fRadius
                pfP2[1]=lvaPeaks2[j].y -fRadius
                pfP2[2]=lvaPeaks2[j].z -fRadius
                pfP2 = src_common.vec3D_norm_3d(pfP2)
                QuickFeat2.interpolate_discrete_3D_point(fioT2,lvaPeaks2[j].x, lvaPeaks2[j].y, lvaPeaks2[j].z)
                QuickFeat2.conv_loc_2_subpixel_prec(-fRadius)
                pfP2[0]=QuickFeat2.x
                pfP2[1]=QuickFeat2.y
                pfP2[2]=QuickFeat2.z
                pfP2=src_common.vec3D_norm_3d(pfP2)
                fParallelMag = src_common.vec3D_dot_3d(pfP1, pfP2)
                assert(np.fabs(fParallelMag)<0.5)
                pfP2 = pfP2 - (fParallelMag*pfP1)
                OriData2.append(QuickFeat2)
                pfP2 = src_common.vec3D_norm_3d(pfP2)
                pfP3 = src_common.vec3D_cross_3d(pfP1, pfP2)
                fullList=np.concatenate((pfP1, pfP2, pfP3))
                pfOriMatrix.append(fullList)
                iOrientationsReturned+=1

        return iOrientationsReturned, pfOriMatrix
