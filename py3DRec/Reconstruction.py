import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as nL
from scipy.optimize import leastsq
import pickle
#from scipy import stats
#from scipy.optimize import minimize
import Camera
import time



class clsReconstruction(object):
	"""description of class"""

	#==========================================================
	@staticmethod
	def saveData(X,filename):
		with open(filename,"wb") as f:
			pickle.dump(X,f)
			


	#==========================================================
	@staticmethod
	def loadData(filename):
		with open(filename,"rb") as f:
			X = pickle.load(f)
		return X



	#==========================================================
	@staticmethod
	def getMatchingPoints(file1,file2,kdef,npoints):
		im_1 = cv2.imread(file1,0)
		im_2 = cv2.imread(file2,0)
		k = clsReconstruction.loadData(kdef)

		return clsReconstruction.getMatchingPointsFromObjects(im_1,im_2,k,npoints)




	#==========================================================
	@staticmethod
	def getMatchingPointsFromObjects(image1,image2,kmatrix,npoints):
		im_1 = image1
		im_2 = image2
		k = kmatrix
		 


		#proceed with sparce feature matching
		#for different features, see 
		#http://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html
		
		detector = cv2.AKAZE_create()

		#detector = cv2.BRISK_create()  


		#detector = cv2.ORB_create()


		kp_1, des_1 = detector.detectAndCompute(im_1,None) 
		kp_2, des_2 = detector.detectAndCompute(im_2,None) 

		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(des_1,des_2)

		matches = sorted(matches, key = lambda x:x.distance)

		#for dense matching... find keypoints for each subwindow and compute its descriptors
		#kp_1 = orb.detect(im_1)
		#des_1 = orb.compute(im_1,kp_1)
		#step = 10
		#keypoints = []
		#for i in range(0,im_1.shape[0],step):
		#	for j in range(0,im_2.shape[1],step):
		#		pass
	
		matches = matches[0:npoints]

		#draw_params = dict(matchColor = (20,20,20), singlePointColor = (250,250,250),
		#			matchesMask = None,
		#			flags = 0)
		#im_3 = cv2.drawMatches(im_1,kp_1,im_2,kp_2,matches[0:npoints], None, **draw_params)
		#plt.imshow(im_3)
		#plt.show()

		pts1 = []
		pts2 = []
		idx =  matches[1:npoints]

		for i in idx:
			pts1.append(kp_1[i.queryIdx].pt)
			pts2.append(kp_2[i.trainIdx].pt)

		return np.array(pts1), np.array(pts2)




	#==========================================================
	@staticmethod
	def sparceRecostructionTrueCase(file1,file2,kdef):

		k = np.mat(clsReconstruction.loadData(kdef))

		#k[0,2] = 0
		#k[1,2] = 0

		ki = np.linalg.inv(k)

		im_1 = cv2.imread(file1)
		im_2 = cv2.imread(file2)

		im_b1 = cv2.cvtColor(im_1,cv2.COLOR_RGB2GRAY)
		im_b2 = cv2.cvtColor(im_2,cv2.COLOR_RGB2GRAY)

		myC1 = Camera.myCamera(k)
		myC2 = Camera.myCamera(k)


		#place camera 1 at origin
		myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])


		#return macthing points
		Xp_1, Xp_2 = clsReconstruction.getMatchingPoints(file1,file2,kdef,40)


		#evaluate the essential Matrix using the camera parameter(using the original points)
		E, mask0 = cv2.findEssentialMat(Xp_1,Xp_2,k,cv2.FM_RANSAC)


		#evaluate Fundamental to get the epipolar lines
		#since we already know the camera intrincics, it is better to evaluate F from the correspondence rather than from the 8 points routine
		F = ki.T*np.mat(E)*ki

		 
		#retrive R and t from E
		retval, R, t, mask2 = cv2.recoverPose(E,Xp_1,Xp_2)
		

		#place camera 2
		myC2.projectiveMatrix(np.mat(t),R)


		#clsReconstruction.drawEpipolarLines(Xp_1,Xp_2,F,im_1,im_2)


		#triangulate points
		Str_4D = cv2.triangulatePoints(myC1.P[:3],myC2.P[:3],Xp_1.transpose()[:2],Xp_2.transpose()[:2]).T


		#make them euclidian
		Str_3D = cv2.convertPointsFromHomogeneous(Str_4D).reshape(-1,3)


		#evaluate reprojection
		Xh_reprojection_1 = myC1.project(Str_4D)
		Xh_reprojection_2 = myC2.project(Str_4D)


		#three ways to carry on the bundle adjustment I am using R,t and K as parameters. using the points is too time 
		# consuming although the results are much better; 
		#nR,nt, R0, R1 = clsReconstruction.bundleAdjustment(Str_4D,Xp_1,Xp_2,k,R,t)
		#Str_4D, nR,nt, R0, R1 = clsReconstruction.bundleAdjustmentwithX(Str_4D,Xp_1,Xp_2,k,R,t)	#### not working right now... 

		nk, nR, nt, R0, R1 = clsReconstruction.bundleAdjustmentwithK(Str_4D,Xp_1,Xp_2,k,R,t)
		print('old value {0:.3f}, optimized pose: {1:.3f} \n'.format(R0,R1))
		nki = np.linalg.inv(nk)


		#reevaluate essential and fundamental matrixes
		nE = clsReconstruction.skew(nt)*np.mat(nR)
		nF = nki.T*np.mat(nE)*nki


		#if we use the 3th option, we should reinitiate the cameras	and the essential matrix, once the projective matrix will change
		myC1 = Camera.myCamera(nk)
		myC2 = Camera.myCamera(nk)
		myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])



		#reevaluate all variables based on new values of nR and nt
		myC2.projectiveMatrix(np.mat(nt),nR)
		Str_4D = cv2.triangulatePoints(myC1.P[:3],myC2.P[:3],Xp_1.transpose()[:2],Xp_2.transpose()[:2]).T
		Str_3D = cv2.convertPointsFromHomogeneous(Str_4D).reshape(-1,3)


		
		#Camera.myCamera.show3Dplot(Str_3D)
		Xh_Opt_1 = myC1.project(Str_4D)#.reshape(-1,2)
		Xh_Opt_2 = myC2.project(Str_4D)#.reshape(-1,2)


		#POSSIBLE IMPLEMENTATION find residuals bigger a threshould value and optimize their location in R3
		#evaluate 




		#clsReconstruction.drawEpipolarLines(Xp_1,Xp_2,nF,im_b1,im_b2)

		im = clsReconstruction.drawPoints(im_1,Xp_1,(50,50,250))
		im = clsReconstruction.drawPoints(im,Xh_reprojection_1,(50,150,100))
		im = clsReconstruction.drawPoints(im,Xh_Opt_1,(250,250,50))

		im2 = clsReconstruction.drawPoints(im_2,Xp_2,(50,50,250))
		im2 = clsReconstruction.drawPoints(im2,Xh_reprojection_2,(50,150,100))
		im2 = clsReconstruction.drawPoints(im2,Xh_Opt_2,(250,250,50))


		cv2.imshow("im",im)
		cv2.imshow("im2",im2)
		cv2.waitKey(0)



	#==========================================================
	@staticmethod
	def bundleAdjustmentwithK(Str_4D, Xp_1, Xp_2, k, R, t):
		#Camera.myCamera.show3Dplot(Xp_3D)
		r_euclidian,jac  = cv2.Rodrigues(R)
		x = np.vstack((r_euclidian,t)).reshape(-1)
		kstack = np.array(k).reshape(-1)

		Xstk = np.hstack((kstack[0:9],x)).reshape(-1)
		

		Res = clsReconstruction.reProjectResidualwithK(Xstk, Str_4D, Xp_1, Xp_2)

		p = nL.optimize.minimize(clsReconstruction.reProjectResidualwithK,Xstk, args = (Str_4D, Xp_1, Xp_2))
		
		nx = np.array(p.x)
		
		
		nRes = clsReconstruction.reProjectResidualwithK(nx, Str_4D, Xp_1, Xp_2, k)

		stackedK = nx[0:9]
		nk = stackedK.reshape(3,3)
		#nk = np.vstack((nk,[0,0,1]))
		x = nx[9:9+6]
	

		nR = cv2.Rodrigues(x[0:3])[0]
		nt = np.array(x[3:6]).reshape(3,1)


		return nk,nR,nt,Res,nRes




	#==========================================================
	@staticmethod
	def bundleAdjustmentwithX(Str_4D, Xp_1, Xp_2, k, R, t):

		shp = Str_4D.shape
		Xstk = Str_4D.reshape(-1)
		#Xstk = np.hstack((xest,x)).reshape(-1)

		Res = clsReconstruction.reProjectResidualwithX(Xstk, shp, Xp_1, Xp_2, k, R, t)


		p = nL.optimize.minimize(clsReconstruction.reProjectResidualwithX,Xstk, args = (shp, Xp_1, Xp_2, k, R, t))
		
		nx = np.array(p.x)
		
		nRes = clsReconstruction.reProjectResidualwithX(nx, shp, Xp_1, Xp_2, k, R, t)

		stackedx = nx[0:shp[0]*shp[1]]
		Str_4D = stackedx.reshape(shp)


		return Str_4D,Res,nRes



	#==========================================================
	@staticmethod
	def bundleAdjustment(Str_4D, Xp_1, Xp_2, k, R, t):
		#Camera.myCamera.show3Dplot(Xp_3D)
		r_euclidian,jac  = cv2.Rodrigues(R)
		x = np.vstack((r_euclidian,t)).reshape(-1)

		Res = clsReconstruction.reProjectResidual(x, Str_4D, Xp_1, Xp_2, k)

		p = nL.optimize.minimize(clsReconstruction.reProjectResidual,x, args = (Str_4D, Xp_1, Xp_2, k))
		
		nx = np.array(p.x)
		
		nRes = clsReconstruction.reProjectResidual(nx, Str_4D, Xp_1, Xp_2, k)

		nR = cv2.Rodrigues(nx[0:3])[0]
		nt = np.array(nx[3:6]).reshape(3,1)

		return nR,nt,Res,nRes




	#==========================================================
	@staticmethod
	def reProjectResidualwithK(nx, *args):
		
		
		Str_4D = args[0]
		Xp_1 = args[1]
		Xp_2 = args[2]

		kstk = nx[0:9]
		x = nx[9:9+6]

		k = kstk.reshape(3,3)
		#k = np.vstack((k,[0,0,1]))

		R = cv2.Rodrigues(x[0:3])
		t = np.array(x[3:6]).reshape(3,1)
		
		myC1 = Camera.myCamera(k)
		myC2 = Camera.myCamera(k)
		myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])
		myC2.projectiveMatrix(np.mat(t),R[0])

		rXp_1 = np.mat(myC1.project(Str_4D))
		rXp_2 = np.mat(myC2.project(Str_4D))
		res_1 = Xp_1 - rXp_1
		res_2 = Xp_2 - rXp_2

		Res = np.hstack((res_1,res_2)).reshape(-1)

		nRes = 2*np.sqrt(np.sum(np.power(Res,2))/len(Res))


		return nRes


	#==========================================================
	@staticmethod
	def reProjectResidualwithX(nx, *args):
		
		shp = args[0]
		Xp_1 = args[1]
		Xp_2 = args[2]
		k = args[3]
		R = args[4]
		t = args[5]

		stackedx = nx[0:shp[0]*shp[1]]
		#x = nx[shp[0]*shp[1]:shp[0]*shp[1]+6]

		Str_4D = stackedx.reshape(shp)

		#R = cv2.Rodrigues(x[0:3])
		#t = np.array(x[3:6]).reshape(3,1)
		
		myC1 = Camera.myCamera(k)
		myC2 = Camera.myCamera(k)
		myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])
		myC2.projectiveMatrix(np.mat(t),R[0])

		rXp_1 = np.mat(myC1.project(Str_4D))
		rXp_2 = np.mat(myC2.project(Str_4D))
		res_1 = Xp_1 - rXp_1
		res_2 = Xp_2 - rXp_2

		Res = np.hstack((res_1,res_2)).reshape(-1)

		nRes = 2*np.sqrt(np.sum(np.power(Res,2))/len(Res))


		return nRes



	#==========================================================
	@staticmethod
	def reProjectResidual(x, *args):
							
		Str_4D = args[0]
		Xp_1 = args[1]
		Xp_2 = args[2]
		k = args[3]

		R = cv2.Rodrigues(x[0:3])
		t = np.array(x[3:6]).reshape(3,1)
		
		myC1 = Camera.myCamera(k)
		myC2 = Camera.myCamera(k)
		myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])
		myC2.projectiveMatrix(np.mat(t),R[0])

		rXp_1 = np.mat(myC1.project(Str_4D))
		rXp_2 = np.mat(myC2.project(Str_4D))
		res_1 = Xp_1 - rXp_1
		res_2 = Xp_2 - rXp_2

		Res = np.hstack((res_1,res_2)).reshape(-1)

		nRes = 2*np.sqrt(np.sum(np.power(Res,2))/len(Res))
		


		return nRes


	#==========================================================
	def drawlines(img1,img2,lines,pts1,pts2):
		''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
		r,c = img1.shape

		img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
		img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

		for r,pt1,pt2 in zip(lines,pts1,pts2):
			color = tuple(np.random.randint(0,255,3).tolist())
			x0,y0 = map(float, [0, -r[2]/r[1] ])
			x1,y1 = map(float, [c, -(r[2]+r[0]*c)/r[1] ])
			img1 = cv2.line(img1, (int(x0),int(y0)), (int(x1),int(y1)), color,1)
			img1 = cv2.circle(img1,tuple((int(pt1[0]),int(pt1[1]))),5,color,-1)
			img2 = cv2.circle(img2,tuple((int(pt2[0]),int(pt2[1]))),5,color,-1)
		return img1,img2
	

	#==========================================================		
	@staticmethod	
	def drawPoints(img1,pts1,color):
		''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
		for pt1 in pts1:
			img1 = cv2.circle(img1,(int(pt1[0]), int(pt1[1])),4,color,2,1)
		return img1			
		

	#==========================================================
	def doSubPlot(plt, position, img, color_argument, title):
		try:
			plt.subplot(position),plt.imshow(cv2.cvtColor(img, color_argument)),plt.title(title)
			plt.xticks([]), plt.yticks([])
		except:
			pass
		return plt		
		
		
		

	@staticmethod
	def skew(v):
		if len(v) == 4: v = v[:3]/v[3]
		skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
		return np.mat(skv - skv.T)



	@staticmethod
	def drawEpipolarLines(Xp_1,Xp_2,F,im_1,im_2):
		F = np.mat(F)
		#get epipolar lines
		lines1 = cv2.computeCorrespondEpilines(Xp_2.reshape(-1,1,2), 2,F)
		lines1 = lines1.reshape(-1,3)
		img3,img4 = clsReconstruction.drawlines(im_1,im_2,lines1,Xp_1,Xp_2)

		lines2 = cv2.computeCorrespondEpilines(Xp_1.reshape(-1,1,2), 1,F)
		lines2 = lines2.reshape(-1,3)
		img5, img6 = clsReconstruction.drawlines(im_2, im_1, lines2,Xp_2, Xp_1)

		plt.subplot(121),plt.imshow(img3)
		plt.subplot(122),plt.imshow(img5)
		plt.show()
		
		
		

	
	
			
		##for dense matching = not using right now
		#ipt1 = matches[0].queryIdx
		#pt1 = kp_1[ipt1]
		#ip1 = (int(pt1.pt[0]), int(pt1.pt[1]))
		#ipt2 = matches[0].trainIdx
		#pt2 = kp_2[ipt2]
		#ip2 = (int(pt2.pt[0]), int(pt2.pt[1]))

		#delta = 10
		#cut1 = im_1[ip1[1]-delta:ip1[1]+delta,ip1[0]-delta:ip1[0]+delta] 
		#cut2 = cut1 * 0

		##calculo da entropia = baixa entropia significa pouca informacao para casar as imagens
		##areas com baixa entropia ou deverao ser ignoradas ou entram com pouco peso e os pixels
		##serao interpolados linearmente, mas sem peso no processo de homografia
		#pp = np.cov(cut1)
		#a,b = np.linalg.eig(pp)

		#pp2 = np.cov(cut2)
		#aa,bb = np.linalg.eig(pp2)
		
		#plt.imshow(cv2.cvtColor(cut1,cv2.COLOR_GRAY2RGB))
		#plt.show()

		##aqui visualiza os pontos capturados nas imagens
		##cv2.circle(im_1,(int(pt1.pt[0]), int(pt1.pt[1])),int(4),(250,50,250),4,2)
		##cv2.circle(im_2,(int(pt2.pt[0]), int(pt2.pt[1])),int(4),(250,50,250),4,2)


		##cv2.imshow('compilado',img3)
		#plt.figure()
		#plt.imshow(img3)
		#plt.show()
		##cv2.imshow('trilho',im)

		##kp = fast.detect(im)
		##im2 = cv2.drawKeypoints(im_1, kp_1, im2, color=(255,0,0))
		##cv2.imshow('trilho22',im2)
		##cv2.imshow('trilho3',im2)
		#cv2.waitKey(0)
