import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class myCamera(object):
	"""description of class"""

	def __init__(self,k):
		""" Initialize P = K[R|t] camera model. """
		self.P = None	  # 
		self.K = np.mat(k) # calibration matrix
		self.R = None # rotation
		self.t = None # translation

	def project(self,X):
		""" Project points in X (4*n array) and normalize coordinates. """
		x = np.dot(self.P,X.T)
		for i in range(3):
			x[i] /= x[2]

		return x
	
	def projectiveMatrix(self,t,r):
		t = np.mat(t)
		if np.array(r).ndim == 1:
			R = np.mat(myCamera.rotationQxyz(r))
		else:
			R = np.mat(r)

		M = np.hstack((R,-R*t))
		M = np.mat(np.vstack((M,[0, 0, 0, 1])))
		I = np.mat(np.hstack((np.diag([1,1,1]),[[0],[0],[0]])))
		self.R  = R
		self.t = t
		self.P = self.K * I * M


	def normalizePoints(self,X):
		mx = np.mean(X, axis = 0)
		Tr = np.mat([[1, 0, -mx[0,0]],[0,1,-mx[0,1]],[0,0,1]])

		xt = (Tr * X.transpose()).transpose()

		d = []
		for i in range(0,len(X)):
			d.append(np.sqrt(xt[i]*xt[i].T))

		s = np.sqrt(2)/max(d)[0,0]

		Sr = np.mat(([s,0,0],[0,s,0],[0,0,1]))

		Xn = np.mat((Sr * xt.T).T)

		Tr = Sr * Tr
	
		return Xn, Tr 
		

	def rotationQx(x):
		Qx = np.mat([[1,0,0],[0, np.cos(x), np.sin(x)],[0, -np.sin(x), np.cos(x)]])
		return Qx
	
	def rotationQy(y):
		Qy = np.mat([[np.cos(y), 0, -np.sin(y)],[0,1,0],[np.sin(y), 0, np.cos(y)]])
		return Qy
	
	def rotationQz(z):
		Qz = np.mat([[np.cos(z), np.sin(z) ,0],[-np.sin(z), np.cos(z), 0],[0, 0, 1]])
		return Qz		

	def rotationQxyz(c):
		return myCamera.rotationQx(c[0])*myCamera.rotationQy(c[1])*myCamera.rotationQz(c[2])

	def rotation_matrix(a):
		""" Creates a 3D rotation matrix for rotation
		around the axis of the vector a. """
		R = np.eye(4)
		R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
		return R

	def showProjectiveView(x,ptstyle):
		plt.plot(x[:,0],x[:,1],ptstyle)
		plt.show()
		pass

	def show3Dplot(x):
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(xs=x[:,2], ys=x[:,0], zs=x[:,1], zdir='z', label='zdir=z')
		plt.show()
		pass

