import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats

import Camera



class clsReconstruction(object):
	"""description of class"""

	@staticmethod
	def saveData(X,filename):
		with open(filename,"wb") as f:
			pickle.dump(X,f)



	@staticmethod
	def loadData(filename):
		with open(filename,"rb") as f:
			X = pickle.load(f)
		return X




	@staticmethod
	def getMathingPoints(file1,file2,kdef,npoints):
		im_1 = cv2.imread(file1,0)
		im_2 = cv2.imread(file2,0)

		##convert to gray
		#im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
		#im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)

		k = clsReconstruction.loadData(kdef)

		return clsReconstruction.getMatchingPointsFromObjects(im_1,im_2,k,npoints)





	@staticmethod
	def getMatchingPointsFromObjects(image1,image2,kmatrix,npoints):
		im_1 = image1
		im_2 = image2
		k = kmatrix
		  
		#proceed with sparce feature matching
		orb = cv2.ORB_create()
		kp_1, des_1 = orb.detectAndCompute(im_1,None)
		kp_2, des_2 = orb.detectAndCompute(im_2,None)
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = bf.match(des_1,des_2)

		matches = sorted(matches, key = lambda x:x.distance)

		#surf = cv2.xfeatures2d.SURF_create()
		#kp_1, des_1 = surf.detectAndCompute(im_1,None)
		#kp_2, des_2 = surf.detectAndCompute(im_2,None)

		#bf = cv2.BFMatcher()
		#matches = bf.knnMatch(des_1,des_2,k=2)
		#good = []
		#for m,n in matches:
		#	if m.distance < 0.4*n.distance:
		#		good.append([m])

		#img3 = cv2.drawMatchesKnn(im_1,kp_1,im_2,kp_2,good,im_1,flags=2)

		#plt.imshow(img3),plt.show()

		

				
		##select points to evaluate the fundamental matrix
		#Pts1 = []
		#Pts2 = []
		#Tg = []
		#dx = im_1.shape[1]

		#for i in matches:
		#	p1 = kp_1[i.queryIdx].pt
		#	p2 = kp_2[i.trainIdx].pt
		#	Pts1.append(p1)
		#	Pts2.append(p2)
		#	tg = np.arctan2(p2[0] - p1[0],dx + p2[1] - p1[1])
		#	Tg.append(tg)

		#		#get the grid to project onto
		#x_grid = np.linspace(-np.pi/3, np.pi/3, 90)
		#vTg = np.array(Tg)
		##evaluate the KDEpdf
		#kde_pdf = stats.gaussian_kde(vTg).evaluate(x_grid)
		#xmax = x_grid[kde_pdf.argmax()]



		matches = matches[0:npoints]
		#matches = sorted(newMaches, key = lambda x: np.arctan2( kp_2[x.trainIdx].pt[0] - kp_1[x.queryIdx].pt[0],dx +  kp_2[x.trainIdx].pt[1] - kp_1[x.queryIdx].pt[1]))

		draw_params = dict(matchColor = (20,20,20), singlePointColor = (200,200,200),
					matchesMask = None,
					flags = 0)
		
		
		im_3 = cv2.drawMatches(im_1,kp_1,im_2,kp_2,matches[0:npoints], None, **draw_params)
		plt.imshow(im_3)
		plt.show()

		#plt.figure()
		#plt.plot(x_grid,kde_pdf)
		#plt.show()
		

		pts1 = []
		pts2 = []
		idx =  matches[1:npoints]

		for i in idx:
			pts1.append(kp_1[i.queryIdx].pt)
			pts2.append(kp_2[i.trainIdx].pt)

		return np.array(pts1), np.array(pts2)







	@staticmethod
	def sparceRecostructionTestCase():

		k = clsReconstruction.loadData('k_cam_hp.dat')

		pt =clsReconstruction.loadData('pt_test.dat')  

		pth = np.mat(np.hstack((pt,np.ones((len(pt),1)))))

		myC1 = Camera.myCamera(k)
		myC1.projectiveMatrix(np.mat([0,10,10]).transpose(),[0, 0, 0])

		myC2 = Camera.myCamera(k)
		myC2.projectiveMatrix(np.mat([129.4095, 250.0000, 66.9873]).transpose(),[np.pi/6.0,-np.pi/12.0,0])


		Xh_1 = np.mat(myC1.project(pth)).transpose()
		Xh_2 = np.mat(myC2.project(pth)).transpose()

		#PLOT IF YOU WHAT TO VISUALIZE THE POINTS IN SPACE AND IN PROJECTIVE VIEWS OF CAMERAS 1 AND 2
		#Camera.myCamera.show3Dplot(pt)
		#Camera.myCamera.showProjectiveView(Xh_1,'-r')
		#Camera.myCamera.showProjectiveView(Xh_2,'-b')

		Xp_1 = np.hstack((Xh_1[:,0], Xh_1[:,1]))
		Xp_2 = np.hstack((Xh_2[:,0], Xh_2[:,1]))



		myC1 = Camera.myCamera(k)
		myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])


		#retorna pontos correspondentes
		Xp_1, Xp_2 = clsReconstruction.getMathingPoints('b4.jpg','b5.jpg','k_cam_hp.dat')


		#evaluate the essential Matrix using the camera parameter(using the original points)
		E, mask0 = cv2.findEssentialMat(Xp_1,Xp_2,k,cv2.FM_RANSAC)

		#evaluate the fundamental matrix (using the normilized points)
		#F, mask = cv2.findFundamentalMat(Xp_1,Xp_2,cv2.FM_RANSAC)	
		#ki = np.linalg.inv(k)

		#R1, R2, t = cv2.decomposeEssentialMat(E)
 

		retval, R, t, mask2 = cv2.recoverPose(E,Xp_1,Xp_2)

		myC2 = Camera.myCamera(k)
		myC2.projectiveMatrix(np.mat(t),R)


		Xp_4Dt = cv2.triangulatePoints(myC1.P[:3],myC2.P[:3],Xp_1.transpose()[:2],Xp_2.transpose()[:2])

		#Xp_4Dt = cv2.triangulatePoints(myC1.P[:3],myC2.P[:3],Xh_1.transpose()[:2],Xh_2.transpose()[:2])

		Xp_4D = Xp_4Dt.T

		for i in range(0,len(Xp_4D)):
			Xp_4D[i] /= Xp_4D[i,3]

		Xp_3D = Xp_4D[:,0:3]

		Camera.myCamera.show3Dplot(Xp_3D)

		Xh_1 = np.mat(myC1.project(Xp_4D)).transpose()


		im = clsReconstruction.drawPoints(cv2.imread('b4.jpg'),Xh_1,'b')

		cv2.imshow("im",im)
		cv2.waitKey(0)



	@staticmethod
	def sparceRecostructionTrueCase(file1,file2,kdef):
		k = np.mat(clsReconstruction.loadData(kdef))
		ki = np.linalg.inv(k)

		im_1 = cv2.imread(file1)
		im_2 = cv2.imread(file2)

		myC1 = Camera.myCamera(k)
		myC2 = Camera.myCamera(k)

		#place camera 1 at origin
		myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])


		#retorna pontos correspondentes
		Xp_1, Xp_2 = clsReconstruction.getMathingPoints(file1,file2,kdef,30)

		#evaluate the essential Matrix using the camera parameter(using the original points)
		E, mask0 = cv2.findEssentialMat(Xp_1,Xp_2,k,cv2.FM_RANSAC)

		#evaluate Fundamental to get the epipolar lines
		#F, mask = cv2.findFundamentalMat(Xp_1,Xp_2,cv2.FM_RANSAC)	
		F = ki.T*np.mat(E)*ki


		#implement the optimal triangulation method to correct matching placement- MVG page 318
		Xp_1, Xp_2 = cv2.correctMatches(F,np.array([Xp_1]),np.array([Xp_2]))
		Xp_1 = Xp_1.reshape(-1,2)
		Xp_2 = Xp_2.reshape(-1,2)

		 
		#retrive R and t from E
		retval, R, t, mask2 = cv2.recoverPose(E,Xp_1,Xp_2)
		
		#place camera 2
		myC2.projectiveMatrix(np.mat(t),R)


		#clsReconstruction.drawEpipolarLines(Xp_1,Xp_2,F,im_1,im_2)

		#triangulate points
		Xp_4D = cv2.triangulatePoints(myC1.P[:3],myC2.P[:3],Xp_1.transpose()[:2],Xp_2.transpose()[:2]).T


		#make them euclidian
		Xp_3D = cv2.convertPointsFromHomogeneous(Xp_4D).reshape(-1,3)

		#plot 3d points
		#Camera.myCamera.show3Dplot(Xp_3D)

		#now we are going to test if the 3D points are acurate in space, by reprojecting them
		#into the perspective plane of camera 1 and camera 2
		rXp_1 = np.mat(myC1.project(Xp_4D))
		rXp_2 = np.mat(myC2.project(Xp_4D))
		res_1 = Xp_1 - rXp_1
		res_2 = Xp_2 - rXp_2

		#(Res_1,Res_2) = clsReconstruction.reProjectResidual(Xp_4D,Xp_1,Xp_2, myC1.K, myC2.R, myC2.t)

		im = clsReconstruction.drawPoints(cv2.imread(file1),Xh_1,'b')

		#cv2.imshow("im",im)
		#cv2.waitKey(0)



	@staticmethod
	def reProjectResidual(Xpoints,Xp_1, Xp_2, k, R, t):

		myC1 = Camera.myCamera(k)
		myC2 = Camera.myCamera(k)

		#place camera 1 at origin
		myC1.projectiveMatrix(np.mat([0,0,0]).transpose(),[0, 0, 0])
		myC2.projectiveMatrix(np.mat(t),R)
		r0 = [0,0,0];
		t0 = [0,0,0];

		x1 = np.dot(myC1.P,Xpoints.T)
		for i in range(3):
			x1[i] /= x1[2]
		x1 = np.mat(x1.T[:,0:2])

		x2 = np.dot(myC2.P,Xpoints.T)
		for i in range(3):
			x2[i] /= x2[2]
		x2 = np.mat(x2.T[:,0:2])

		Xr1 = Xp_1 - x1
		Xr2 = Xp_2 - x2

		Res_1 = np.linalg.norm(Xr1,axis = 1)
		Res_2 = np.linalg.norm(Xr2,axis = 1)
		

		return Res_1, Res_2


	@staticmethod
	def matchingTests(): 
		im_1 = cv2.imread('c1.bmp')
		im_2 = cv2.imread('c2.bmp')

		#k = np.mat(([[ 683.39404297,    0.        ,  267.21336591], [   0.        ,  684.3449707 ,  218.56421036],  [   0.        ,    0.        ,    1.        ]]))

		k = clsReconstruction.loadData('k_cam_hp.dat')
		#resise, if it is necessary
		#im_1 = cv2.resize(im_1,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
		#im_2 = cv2.resize(im_2,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

		#convert to gray
		im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
		im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)


		#proceed with sparce feature matching
		orb = cv2.ORB_create()
		
		kp_1, des_1 = orb.detectAndCompute(im_1,None)
		kp_2, des_2 = orb.detectAndCompute(im_2,None)

		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		matches = bf.match(des_1,des_2)
		
		matches = sorted(matches, key = lambda x:x.distance)

		draw_params = dict(matchColor = (20,20,20), singlePointColor = (200,200,200),
							matchesMask = None,
							flags = 0)

		
		im_3 = cv2.drawMatches(im_1,kp_1,im_2,kp_2,matches[0:20], None, **draw_params)
		
		
		#select points to evaluate the fundamental matrix
		pts1 = []
		pts2 = []
		idx =  matches[1:20]

		for i in idx:
			pts1.append(kp_1[i.queryIdx].pt)
			pts2.append(kp_2[i.trainIdx].pt)
	

		
		pts1 = np.array(pts1)
		pts2 = np.array(pts2)

		#creating homegeneous coordenate
		pones = np.ones((1,len(pts1))).T

		pth_1 = np.hstack((pts1,pones))
		pth_2 = np.hstack((pts2,pones))

		k = np.array(k)
		ki = np.linalg.inv(k)
		#normalized the points
		pthn_1 = []
		pthn_2 = []

		for i in range(0,len(pts1)):
			pthn_1.append((np.mat(ki) * np.mat(pth_1[i]).T).transpose())
			pthn_2.append((np.mat(ki) * np.mat(pth_2[i]).T).transpose())

		ptn1 = []
		ptn2 = []
		for i in range(0,len(pts1)):
			ptn1.append([pthn_1[i][0,0],pthn_1[i][0,1]])
			ptn2.append([pthn_2[i][0,0],pthn_2[i][0,1]])

		ptn1 = np.array(ptn1)
		ptn2 = np.array(ptn2)
		#evaluate the essential Matrix (using the original points, not the normilized ones)
		E, mask0 = cv2.findEssentialMat(pts1,pts2,k,cv2.FM_RANSAC)
		#evaluate the fundamental matrix (using the normilized points)
		F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)	

		E = np.mat(E)
		F = np.mat(F)


		R, t, mask1 = cv2.recoverPose(E,ptn1,ptn2)
		
		#selecting only inlier points
		ptn1 = ptn1[mask.ravel() == 1]
		ptn2 = ptn2[mask.ravel() == 1]

		# Find epilines corresponding to points in right image (second image) and
		# drawing its lines on left image
		lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
		lines1 = lines1.reshape(-1,3)
		img5,img6 = clsReconstruction.drawlines(im_1,im_2,lines1,pts1,pts2)
		# Find epilines corresponding to points in left image (first image) and
		# drawing its lines on right image
		lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
		lines2 = lines2.reshape(-1,3)
		img3,img4 = clsReconstruction.drawlines(im_2,im_1,lines2,pts2,pts1)
		plt.subplot(131),plt.imshow(img5)
		plt.subplot(132),plt.imshow(img3)
		plt.subplot(133),plt.imshow(im_3)

		plt.show()


	#=======================================
	@staticmethod
	def NewMatching(file1, file2, kdef): 

		k = np.mat(clsReconstruction.loadData(kdef))
		ki = np.linalg.inv(k)
		im_1 = cv2.imread(file1,0)
		im_2 = cv2.imread(file2,0)

		Xp_1, Xp_2 = clsReconstruction.getMatchingPointsFromObjects(im_1,im_2,k)

		d1 = 120
		d2 = 120
		indc = 1;
		idy1 = (max(0,int(Xp_1[indc][1])-d1), min(int(Xp_1[indc][1])+d1,im_1.shape[1]))
		idx1 = (max(0,int(Xp_1[indc][0])-d1), min(int(Xp_1[indc][0])+d1,im_1.shape[0]))
		idy2 = (max(0,int(Xp_2[indc][1])-d2), min(int(Xp_2[indc][1])+d2,im_2.shape[1]))
		idx2 = (max(0,int(Xp_2[indc][0])-d2), min(int(Xp_2[indc][0])+d2,im_2.shape[0]))

		cut_1 = im_1[idy1[0]:idy1[1],idx1[0]:idx1[1]] 
		cut_2 = im_2[idy2[0]:idy2[1],idx2[0]:idx2[1]] 

		#plt.figure()
		#clsReconstruction.doSubPlot(plt,121,cut_1, cv2.COLOR_GRAY2RGB, 'cut 1') 
		#clsReconstruction.doSubPlot(plt,122,cut_2, cv2.COLOR_GRAY2RGB, 'cut 2')
		#plt.show()

		Xcp_1, Xcp_2 = clsReconstruction.getMatchingPointsFromObjects(cut_1,cut_2,k)
		M, mask = cv2.findHomography(Xcp_1, Xcp_2, cv2.RANSAC,5.0)

		

		if M is not None:
			result = cv2.warpPerspective(cut_1, M, (cut_2.shape[1] + cut_2.shape[1], im_2.shape[0]))
			erro = cut_2 - cut_2
			erro[0:cut_2.shape[0], 0:cut_2.shape[1]] = result[0:cut_2.shape[0], 0:cut_2.shape[1]] - cut_2
			result[0:cut_2.shape[0], 0:cut_2.shape[1]] = cut_2
			clsReconstruction.doSubPlot(plt,121,erro, cv2.COLOR_GRAY2RGB, 'parte 1')
			clsReconstruction.doSubPlot(plt,121,result, cv2.COLOR_GRAY2RGB, 'parte 1')
			plt.show()

		

		
	
			 

		matchesMask = mask.ravel().tolist()

		#calculo da entropia = baixa entropia significa pouca informacao para casar as imagens
		#areas com baixa entropia ou deverao ser ignoradas ou entram com pouco peso e os pixels
		#serao interpolados linearmente, mas sem peso no processo de homografia
		#pp = np.cov(cut1)
		#a,b = np.linalg.eig(pp)

		#pp2 = np.cov(cut2)
		#aa,bb = np.linalg.eig(pp2)
		


		#aqui visualiza os pontos capturados nas imagens
		cv2.circle(im_1,(int(pt1.pt[0]), int(pt1.pt[1])),int(4),(250,250,250),4,2)
		cv2.circle(im_2,(int(pt2.pt[0]), int(pt2.pt[1])),int(4),(250,50,250),4,2)


		#cv2.imshow('compilado',img3)
		#cv2.imshow('pt1',im_1)
		#cv2.imshow('pt2',im_2)
		#cv2.waitKey(0)
		#plt.figure()
		#plt.imshow(img3)
		#plt.show()
		#cv2.imshow('trilho',im)

		#plt.imshow(cv2.cvtColor(cut1,cv2.COLOR_GRAY2RGB))
		plt.figure(1)
		clsReconstruction.doSubPlot(plt,221,im_1, cv2.COLOR_GRAY2RGB, 'original 1')
		clsReconstruction.doSubPlot(plt,222,im_2, cv2.COLOR_GRAY2RGB, 'original 2')
		clsReconstruction.doSubPlot(plt,223,cut_1, cv2.COLOR_GRAY2RGB, 'parte 1')
		clsReconstruction.doSubPlot(plt,224,cut_2, cv2.COLOR_GRAY2RGB, 'parte 2')
		plt.show()

		#kp = fast.detect(im)
		#im2 = cv2.drawKeypoints(im_1, kp_1, im2, color=(255,0,0))
		#cv2.imshow('trilho22',im2)
		#cv2.imshow('trilho3',im2)
		cv2.waitKey(0)


	@staticmethod
	def returnMatching(im_1,im_2, ndraw):
		im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
		im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)
		#cv2.imshow('trilho',im_1)

		orb = cv2.ORB_create()
		fast = cv2.FastFeatureDetector_create()

		kp_1, des_1 = orb.detectAndCompute(im_1,None)
		kp_2, des_2 = orb.detectAndCompute(im_2,None)

		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		matches = bf.match(des_1,des_2)
		
		matches = sorted(matches, key = lambda x:x.distance)

		draw_params = dict(matchColor = (20,20,20), singlePointColor = (200,200,200),
							matchesMask = None,
							flags = 0)
		
		img3 = cv2.drawMatches(im_1,kp_1,im_2,kp_2,matches[0:ndraw], None, **draw_params)

		return kp_1,kp_2, des_1, des_2, matches, img3

	
	@staticmethod	
	def drawlines(img1,lines,pts1):
		''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
		r,c,z = img1.shape
		#img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
		#img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
		for r,pt1 in zip(lines,pts1):
			color = tuple(np.random.randint(0,255,3).tolist())
			x0,y0 = map(int, [0, -r[2]/r[1] ])
			x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
			img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
			img1 = cv2.circle(img1,tuple((int(pt1[0]),int(pt1[1]))),5,color,-1)
		return img1		
		
		
	@staticmethod	
	def drawPoints(img1,pts1,color):
		''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
		for pt1 in pts1:
			img1 = cv2.circle(img1,(int(pt1[0,0]), int(pt1[0,1])),5,(250,250,50),4,2)
		return img1			
		
	#=======================================
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
		return skv - skv.T

	@staticmethod
	def drawEpipolarLines(Xp_1,Xp_2,F,im_1,im_2):
		F = np.mat(F)
		#get epipolar lines
		lines1 = cv2.computeCorrespondEpilines(Xp_2.reshape(-1,1,2), 2,F)
		lines1 = lines1.reshape(-1,3)
		lines2 = cv2.computeCorrespondEpilines(Xp_1.reshape(-1,1,2), 1,F)
		lines2 = lines2.reshape(-1,3)
		img5 = clsReconstruction.drawlines(im_1,lines1,Xp_1)
		img6 = clsReconstruction.drawlines(im_2,lines2,Xp_2)
		plt.subplot(121),plt.imshow(img5)
		plt.subplot(122),plt.imshow(img6)
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
