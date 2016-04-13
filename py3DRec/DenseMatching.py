import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as nL
from heapq import heappush, heappop


class clsDenseMatching(object):
	"""description of class"""


	@staticmethod
	def denseMatching(im1, im2, i_matching, half_size_window):
		
		im_1 = cv2.imread(im1,0)
		im_2 = cv2.imread(im2,0)

		zncc1 = clsDenseMatching.returnZncc(im_1,half_size_window)
		zncc2 = clsDenseMatching.returnZncc(im_2,half_size_window)

		reliable_1 = clsDenseMatching.ReliableArea(im_1)
		reliable_2 = clsDenseMatching.ReliableArea(im_2)


		plt.subplot(211),plt.imshow(reliable_1)
		plt.subplot(212),plt.imshow(reliable_2)
		plt.show()


		match_im_1 = np.zeros((reliable_1.shape[0],reliable_1.shape[1],2))
		match_im_2 = np.zeros((reliable_2.shape[0],reliable_2.shape[1],2))

		match_im_1[:,:,0] = reliable_1 - 2
		match_im_1[:,:,0] = reliable_1

		match_im_2[:,:,0] = reliable_2 - 2
		match_im_2[:,:,0] = reliable_2 

		MaxIndexValidMatch  = reliable_1.shape[0] * reliable_1.shape[1] * 2
		NbMaxStartMatch = MaxIndexValidMatch+5*5*9

		#match_heap = hp.heap(NbMaxStartMatch*25+MaxIndexValidMatch)

		heap = []

		CostMax = 0.5

		match_pair = i_matching 

		for i in range(0,i_matching.shape[0]):
			match_pair[i,4] = np.sum(zncc1[match_pair[i,1],match_pair[i,0],:] * zncc1[match_pair[i,3],match_pair[i,2],:])
			heappush(heap, (match_pair[i,4],i))
			pass

		#while (	maxMatchingNumber >=0  and len(heap) > 0 ):

		j = 2

	@staticmethod
	def propagate(mat1, mat2, im_1, im_2, matchable_im_1, matchable_im_2, zncc_1, zncc_2, WinHalfSize, CostMax = 0.5):
		CostMax = 0.5


		match_im_1 = np.zeros(( im_1.shape[0] ,  im_1.shape[1] ,  2))
		match_im_2 = np.zeros(( im_1.shape[0] ,  im_1.shape[1] ,  2))

		match_im_1[:,:,0] = matchable_im_1 - 2
		match_im_1[:,:,1] = match_im_1[:,:,0] 

		match_im_2[:,:,0] = matchable_im_2 - 2
		match_im_2[:,:,1] = match_im_2[:,:,0] 

		maxMatchingNumber = im_1.shape[0] * im_1.shape[1]
		MaxIndexValidMatch = maxMatchingNumber

		NbMaxStartMatch = MaxIndexValidMatch+5*5*9;

		vzeros = np.zeros((mat1.shape[0],1))
		match_pair = np.hstack((mat1,mat2,vzeros[:]))

		heap = []

		match_pair_size = 0

		for i in range(0,mat1.shape[0]):
			ma = np.array(zncc_1[match_pair[i,1],match_pair[i,0],:])
			mb = np.array(zncc_1[match_pair[i,3],match_pair[i,2],:])
			match_pair[i,4] =sum(ma * mb)
			heappush(heap,(-match_pair[i,4],i))
	
			
		while (	maxMatchingNumber >= 0 and len(heap) > 0 ):
			item = heappop(heap)

			x0 = match_pair[item[1],0]
			y0 = match_pair[item[1],1]

			x1 = match_pair[item[1],2]
			y1 = match_pair[item[1],3]

			xMin0= max(WinHalfSize+1, x0-WinHalfSize);     
			xMax0= min(matchable_im_1.shape[1]-WinHalfSize, x0+WinHalfSize+1);
			yMin0= max(WinHalfSize+1, y0-WinHalfSize);     
			yMax0= min(matchable_im_1.shape[0]-WinHalfSize, y0+WinHalfSize+1);

			xMin1= max(WinHalfSize+1, x1-WinHalfSize);     
			xMax1= min(matchable_im_2.shape[1]-WinHalfSize, x1+WinHalfSize+1);
			yMin1= max(WinHalfSize+1, y1-WinHalfSize);     
			yMax1= min(matchable_im_2.shape[0]-WinHalfSize, y1+WinHalfSize+1);


			local_heap = []
			for yy0 in range(int(yMin0),int(yMax0)):
				for	xx0 in range(int(xMin0),int(xMax0)):
					if (match_im_1[yy0,xx0,0] == -1):
						xx = (xx0 + x1) - x0
						yy = (yy0 + y1) - y0
						for yy1 in range(int(max(yMin1,yy-1)),int(min(yMax1,yy+2))):
							for xx1 in range(int(max(xMin1,xx-1)),int(min(xMax1,xx+2))):
								if (match_im_2[yy1,xx1,0] == -1):
									AuxCost= sum(np.array(zncc_1[yy0,xx0,:]) * np.array(zncc_2[yy1,xx1,:]))
									if (1 - AuxCost <= CostMax):
										local_heap.append([xx0,yy0,xx1,yy1,AuxCost])


			
			if len(local_heap) > 0:
				local_heap.sort(key=lambda x: x[4],reverse = True)
				for bestIndex in range(0,len(local_heap)):
					xx0 = local_heap[bestIndex][0]
					yy0 = local_heap[bestIndex][1]
					xx1 = local_heap[bestIndex][2]
					yy1 = local_heap[bestIndex][3]

					
					if (match_im_1[yy0,xx0,0] < 0 and match_im_1[yy0,xx0,1] < 0 and match_im_2[yy1,xx1,0] < 0 and match_im_2[yy1,xx1,1] < 0):
						match_im_1[yy0,xx0,:] = [yy0,yy0]
						match_im_2[yy1,xx1,:] = [xx1,yy1]
						match_pair = np.vstack((match_pair,local_heap[bestIndex]))
						match_pair_size = len(match_pair) -1 
						heappush(heap,(-local_heap[bestIndex][4],match_pair_size))
						
						maxMatchingNumber = maxMatchingNumber - 1




		return match_pair, match_im_1, match_im_2
		#heappush(heap, item)
		#item = heappop(heap)
		#maxMatchingNumber = 



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
	def returnZncc(im, half_size_window):
		
		hsw = half_size_window

		zncc = np.zeros(( im.shape[0] - 2 * hsw,  im.shape[1] - 2 * hsw,  (2 * hsw + 1)**2))
		nzncc = np.zeros(( im.shape[0] ,  im.shape[1] ,  (2 * hsw + 1)**2))
		d0 = im.shape[0]  
		d1 = im.shape[1]  

		k = 000

		for x in range(0,2*hsw+1):
			for y in range (0,2*hsw+1):
				zncc[:,:,k] = im[ x :d0+x-2*hsw   ,y :d1+y-2*hsw ]
				zncc[:,:,k] = zncc[:,:,k]/256 
				k = k + 1

		
		zncc_mean = np.mean(zncc,axis = 2)


		sum_deviation = 0
		for i in range(0,(2 * hsw + 1)**2):
			sum_deviation = sum_deviation + np.array((zncc[:,:,i] - zncc_mean))*np.array((zncc[:,:,i] - zncc_mean))

		sum_deviation = np.sqrt(sum_deviation)

		for i in range(0,(2 * hsw + 1)**2):
			zncc[:,:,i] = np.array((zncc[:,:,i] - zncc_mean)) / np.array(sum_deviation)
			nzncc[hsw:-hsw,hsw:-hsw,i] = zncc[:,:,i]

			
			
		return nzncc
		

	@staticmethod
	def ReliableArea(im):
		#sb_1 = cv2.Sobel(im,cv2.CV_8U,1,0,ksize=1)
		#sb_2 = cv2.Sobel(im,cv2.CV_8U,0,1,ksize=1)
		a = np.array([(cv2.Sobel(im,cv2.CV_8U,1,0,ksize=1)), (cv2.Sobel(im,cv2.CV_8U,0,1,ksize=1)), (cv2.Sobel(im,cv2.CV_8U,1,1,ksize=1))]).max(axis = 0)
#		a = cv2.Sobel(im,cv2.CV_8U,1,0,ksize=1)
		#a = a/256;
		a[0,:] = 0
		a[-1,:] = 0
		a[:,0] = 0
		a[:,-1] = 0
		thresh = 3
		#im_bw = 255 - cv2.threshold(a, thresh, 255, cv2.THRESH_BINARY)[1]
		im_bw = cv2.threshold(a, thresh, 255, cv2.THRESH_BINARY)[1]
		b = a < 0.01
		b = 1 - b
		return im_bw/255
		
		


