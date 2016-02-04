import numpy as np
import cv2
import matplotlib.pyplot as plt

#import Reconstruction
from Reconstruction import *
import Camera



k = clsReconstruction.loadData('k_cam_hp.dat')

pt =clsReconstruction.loadData('pt_test.dat')  

pth = np.mat(np.hstack((pt,np.ones((len(pt),1)))))

pp=1
#myC1 = Camera.myCamera(k)
#myC1.projectiveMatrix(np.mat([0,10,10]).transpose(),[0, 0, 0])

#myC2 = Camera.myCamera(k)
#myC2.projectiveMatrix(np.mat([129.4095, 250.0000, 66.9873]).transpose(),[np.pi/6.0,-np.pi/12.0,0])


#Xh_1 = np.mat(myC1.project(pth)).transpose()
#Xh_2 = np.mat(myC2.project(pth)).transpose()

#Camera.myCamera.show3Dplot(pt)
#Camera.myCamera.showProjectiveView(Xh_1,'-r')
#Camera.myCamera.showProjectiveView(Xh_2,'-b')

#normaliza os pontos de acordo com a projecao
#Xhn_1, Trh_1 = myC1.normalizePoints(Xh_1)
#Xhn_2, Trh_2 = myC2.normalizePoints(Xh_2)

#Camera.myCamera.showProjectiveView(xn_1,'-r')
#Camera.myCamera.showProjectiveView(xn_2,'-b')

#Xp_1 = np.hstack((Xh_1[:,0], Xh_1[:,1]))
#Xp_2 = np.hstack((Xh_2[:,0], Xh_2[:,1]))



myC1 = Camera.myCamera(k)
myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])


#retorna pontos correspondentes
Xp_1, Xp_2 = clsReconstruction.getMathingPoints('b4.jpg','b5.jpg','k_cam_hp.dat')


#evaluate the essential Matrix using the camera parameter(using the original points)
E, mask0 = cv2.findEssentialMat(Xp_1,Xp_2,k,cv2.FM_RANSAC)

#evaluate the fundamental matrix (using the normilized points)
#F, mask = cv2.findFundamentalMat(Xp_1,Xp_2,cv2.FM_RANSAC)	
#ki = np.linalg.inv(k)

R1, R2, t = cv2.decomposeEssentialMat(E)



retval, R, t, mask2 = cv2.recoverPose(E,Xp_1,Xp_2)

myC2 = Camera.myCamera(k)
myC2.projectiveMatrix(np.mat(t),R)


Xp_4Dt = cv2.triangulatePoints(myC1.P[:3],myC2.P[:3],Xp_1.transpose()[:2],Xp_2.transpose()[:2])

#Xp_4Dt = cv2.triangulatePoints(myC1.P[:3],myC2.P[:3],Xh_1.transpose()[:2],Xh_2.transpose()[:2])

Xp_4D = Xp_4Dt.T

for i in range(0,len(Xp_4D)):
	Xp_4D[i] /= Xp_4D[i,3]

Xp_3D = Xp_4D[:,0:3]

#Camera.myCamera.show3Dplot(Xp_3D)

Xh_1 = np.mat(myC1.project(Xp_4D)).transpose()


im = clsReconstruction.drawPoints(cv2.imread('b4.jpg'),Xh_1,'b')

cv2.imshow("im",im)
cv2.waitKey(0)

#Reconstruction.clsReconstruction.matchingTests();