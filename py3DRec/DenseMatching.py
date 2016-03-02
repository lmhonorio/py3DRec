import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as nL

class clsDenseMatching(object):
	"""description of class"""


	@staticmethod
	def DMZNCC(im1, im2, i_matching, half_size_window):
		
		im_1 = cv2.imread(im1,0)
		im_2 = cv2.imread(im2,0)

		zncc1 = clsDenseMatching.returnZncc(im_1,half_size_window)
		zncc2 = clsDenseMatching.returnZncc(im_2,half_size_window)

		reliable_1 = clsDenseMatching.ReliableArea(im_1)
		reliable_2 = clsDenseMatching.ReliableArea(im_2)




		pass

	@staticmethod
	def propagate(im1, im2, i_matching, half_size_window, zncc1, zncc2, reliable1, reliable2):
		
		im_1 = cv2.imread(im1,0)
		im_2 = cv2.imread(im2,0)

		zncc1 = clsDenseMatching.returnZncc(im_1,half_size_window)
		zncc2 = clsDenseMatching.returnZncc(im_2,half_size_window)

		reliable_1 = clsDenseMatching.ReliableArea(im_1)
		reliable_2 = clsDenseMatching.ReliableArea(im_2)

	@staticmethod
	def returnZncc(im, half_size_window):
		
		hsw = half_size_window

		zncc = np.zeros(( im.shape[0],  im.shape[1],  (2 * hsw + 1)**2))
		d0 = im.shape[0]  
		d1 = im.shape[1]  

		k = 0

		for x in range(0,2*hsw+1):
			for y in range (0,2*hsw+1):
				zncc[hsw:-hsw,hsw:-hsw,k] = im[ x :d0+x-2*hsw   ,y :d1+y-2*hsw ]
				k = k + 1

		zncc_mean = np.mean(zncc,axis = 2)

		zncc_deviation = 0
		for i in range(0,(2 * hsw + 1)**2):
			zncc_deviation += (zncc[:,:,i] - zncc_mean)**2

		zncc_deviation = zncc_deviation**(1/2)

		for i in range(0,(2 * hsw + 1)**2):
			zncc[:,:,i] = (zncc[:,:,i] - zncc_mean)/ zncc_deviation

		return zncc
		

	@staticmethod
	def ReliableArea(im):
		#sb_1 = cv2.Sobel(im,cv2.CV_8U,1,0,ksize=1)
		#sb_2 = cv2.Sobel(im,cv2.CV_8U,0,1,ksize=1)
		a = np.array([(cv2.Sobel(im,cv2.CV_8U,1,0,ksize=1)), (cv2.Sobel(im,cv2.CV_8U,0,1,ksize=1)), (cv2.Sobel(im,cv2.CV_8U,1,1,ksize=1))]).max(axis = 0)
		a = a/ 256
		a[0,:] = 0
		a[-1,:] = 0
		a[:,0] = 0
		a[:,-1] = 0
		b = a < 0.01
		b = 1 - b
		return b
		
		


