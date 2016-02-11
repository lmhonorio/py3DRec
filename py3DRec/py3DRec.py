import numpy as np
import cv2
import matplotlib.pyplot as plt

#import Reconstruction
from Reconstruction import *
import Camera



clsReconstruction.sparceRecostructionTrueCase('b4.jpg','b5.jpg','k_cam_hp.dat')

#clsReconstruction.NewMatching('b4.jpg','b5.jpg','k_cam_hp.dat')