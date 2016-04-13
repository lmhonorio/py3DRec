import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
#import Reconstruction
from Reconstruction import *
import Camera
import time
import DenseMatching





"""3D RECONSTRUCTION.

These routines use code style based by the `Google Python
Style Guide`. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.
for 

variables: lowercase with underscore in case of multiple words, avoid resuming if possible
	im
	im_1
	structure_homogeneous

Properties:	same style as variables

Attributes: all togheter, without underscore and the second word is always capitalized.
    imagemName
	imagePropertyParameters
    imagem

Classes: start with cls and uses UpperAndLower format
	clsParameterMacher

Objetcts: if possible, same name of the class without the cls marker. Always starts with capital case
	ParameterMatcher_<object_sub_name>
	ParameterMatcher_Flann
    

Examples: if possible define some example for routine usage
    obj = clsImageTest.clsImage(<filename>)
    tuple = obj.imageShape


Authors:  Define the name and alias here. any changes from the master branch must be signed by the alias and with date
    (LEO) Leonardo de Mello Honorio 
    (CHV) Carlos Henrique Valerio de Moraes

Last Modified:
    (LEO) : 23/02/2016

Version:
    v0.1


LIBRARIES VERSION USED LIBS: 
		see requirements.txt (automatically generated by visual studio virtual environments)

    

"""

#x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
#res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)



#im = cv2.imread('b21.jpg',0)
#k = DenseMatching.clsDenseMatching.ReliableArea(im)
#cv2.imshow('t',k)
#cv2.waitKey(0)

#im = cv2.imread('b4.jpg',0)
t0 = time.time()
i = 1

#Xp_1, Xp_2, Str_4D = clsReconstruction.sparceRecostructionTrueCase('a8.jpg','a9.jpg','k_cam_hp.dat')

im_1 = cv2.imread('a8.jpg',0)
im_2 = cv2.imread('a9.jpg',0)

im_1 = cv2.resize(im_1,(640,360))
im_2 = cv2.resize(im_2,(640,360))

kdef = clsReconstruction.loadData('k_cam_hp.dat')
Xp_1, Xp_2 = clsReconstruction.getMatchingPointsFromObjects(im_1,im_2,kdef,20)
 
half_size_window = 1

zncc_1 = DenseMatching.clsDenseMatching.returnZncc(im_1,half_size_window)
zncc_2 = DenseMatching.clsDenseMatching.returnZncc(im_2,half_size_window)

matchable_im_1 = DenseMatching.clsDenseMatching.ReliableArea(im_1)
matchable_im_2 = DenseMatching.clsDenseMatching.ReliableArea(im_2)

matches, aa, bb = DenseMatching.clsDenseMatching.propagate(Xp_1, Xp_2, im_1, im_2, matchable_im_1, matchable_im_2, zncc_1, zncc_2, half_size_window, CostMax = 0.5)


xp_1 = np.vstack((matches[:,0],matches[:,1])).T

xp_2 = np.vstack((matches[:,1],matches[:,2])).T

im1 = clsReconstruction.drawPoints(im_1,xp_1,(250,50,50))
im2 = clsReconstruction.drawPoints(im_2,xp_2,(250,50,50))

cv2.imshow('lf',im1)
cv2.imshow('rg',im2)

print(time.time() - t0)
cv2.waitKey(0)
#reliable_1 = DenseMatching.clsDenseMatching.ReliableArea(im_1)
#reliable_2 = DenseMatching.clsDenseMatching.ReliableArea(im_2)
#plt.subplot(121),plt.imshow(cv2.cvtColor(reliable_1,cv2.COLOR_GRAY2RGB))
#plt.subplot(122),plt.imshow(cv2.cvtColor(reliable_2,cv2.COLOR_GRAY2RGB))
#plt.show()




##Xp_1, Xp_2, Str_4D = clsReconstruction.sparceRecostructionTrueCase('b21.jpg','b22.jpg','k_cam_hp.dat')
#Xp_1, Xp_2, Str_4D = clsReconstruction.sparceRecostructionTrueCase('a8.jpg','a9.jpg','k_cam_hp.dat')
#zr = np.zeros((Xp_1.shape[0],1))

#Xmatch = np.hstack((Xp_1,Xp_2,zr))

#DenseMatching.clsDenseMatching.denseMatching('a8.jpg','a9.jpg',Xmatch,1)

