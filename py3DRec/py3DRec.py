import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
#import Reconstruction
from Reconstruction import *
import Camera


x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)



clsReconstruction.sparceRecostructionTrueCase('b4.jpg','b5.jpg','k_cam_hp.dat')

#clsReconstruction.NewMatching('b4.jpg','b5.jpg','k_cam_hp.dat')