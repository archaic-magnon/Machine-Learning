import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import time
from mpl_toolkits.mplot3d import Axes3D
import sys
from helper import *
import math


def loadData(x_file, y_file):
	'''Load data logisticX.csv and logisticY.csv
		Return type: dict
		{'X': X, 'Y' : Y}
	'''
	X = designMatrix(normalize(np.array(pd.read_csv(x_file, sep="\s+", header=None))))
	Y = np.array(pd.read_csv(y_file, header=None))

	return {'X' : X, 'Y': Y}


# φ =1/m Σ1{y(i) = 1}
def phiFunc(Y):
	m = Y.shape[0]
	ones = np.count_nonzero(Y == "Canada")
	return ones / m

def meanFunc(X, Y, S):
	denominator = np.count_nonzero(Y == S)

	sum_x = 0
	for index , row in enumerate(Y):
		if row[0] == S:
			sum_x+= X[index][1]
	u1 =  sum_x / denominator
	
	sum_x = 0
	for index , row in enumerate(Y):
		if row[0] == S:
			sum_x+= X[index][2]
	u2 =  sum_x / denominator
	
	return np.array([[u1], [u2]])


def covarMatrixFunc(X, Y, u0, u1):
	m = X.shape[0]
	n = X.shape[1]
	sigma_Matrix = 0
	for i in range(0, m):
		x_i = (X[i][1:].T).reshape(2,1)
		if Y[i][0] == "Alaska":
			v = x_i - u0
		else:
			v = x_i - u1
			
		sigma_Matrix += np.dot(v, v.T)
	return sigma_Matrix / m 


def getXData(X, Y, val = 0):
	m = X.shape[1]
	new_x = np.empty((0, m))
	for index, row in enumerate(Y):
		if row[0] == val:
			new_x = np.append(new_x, X[index])
	new_x = new_x.reshape((-1,m))   
	return new_x  

def calcCoeffLinear(u0, u1, sig, phi):
	#(u0-u1).T * inv(sigma)
	sig_inv = np.linalg.inv(sig)
	c1 =  np.dot( ((u0-u1).T), sig_inv) 
	
	# a_temp1 = np.dot(u1.T, sig_inv)
	# a_temp2 = np.dot(a_temp1, u1)
	
	# b_temp1 = np.dot(u0.T, sig_inv)
	# b_temp2 = np.dot(b_temp1, u0)
	
	# b_temp3 = b_temp2 - math.log(phi/(1 - phi))
	
	c2 = (1/2) * (u1.T).dot(sig_inv).dot(u1) - (u0.T).dot(sig_inv).dot(u0)

	# c2 = 1/2 *(a_temp2 - b_temp2)
	
	return (c1, c2)


def plotDecisionBoundry(u0, u1, sig, phy, X, Y):
	(A, c) = calcCoeff(u0, u1, sig, phy)
	a1 = A[0][0]
	a2 = A[0][1]
	x1 = X[:, 1].reshape(-1, 1)
	x2 = -(c + a1*x1) / a2
	
	xx0 = getXData(X, Y, val="Alaska")
	x10 = xx0[:,1]
	x20 = xx0[:,2]
	
	xx1 = getXData(X, Y, val="Canada")
	x11 = xx1[:,1]
	x21 = xx1[:,2]
	
	plt.plot(x1, x2, label="Boundary")
	plt.scatter(x10, x20,color="r", label="Alaska", marker="+")
	plt.scatter(x11, x21,color="b", label="Canada", marker="*")
	plt.title("Decision Boundary")
	plt.legend(loc="upper right")
#	 plt.xlabel("x1")
#	 plt.ylabel("x2")
	

def sigmaGeneral(X, Y, u0, u1, value=0):
	m = Y.shape[0]
	y_count = 0
	u = 0

	n = X.shape[1]

	S = ""
	if value == 0:
		S = "Alaska"
		u = u0
	else:
		S = "Canada"
		u = u1

  
	sigma_Matrix = 0
	for i in range(0, m):
		if Y[i][0] == S:
			y_count += 1
			x_i = (X[i][1:].T).reshape(2,1)
			v = x_i - u
			sigma_Matrix += np.dot(v, v.T)

	return sigma_Matrix / y_count


def calcCoeffQuad(sigma0, sigma1, u0, u1, phi):
    sig0_inv = np.linalg.inv(sigma0)
    sig1_inv = np.linalg.inv(sigma1)
    
    Q = sig0_inv - sig1_inv
    
    u_0_s_0 = (u0.T).dot(sig0_inv)
    u_1_s_1 = (u1.T).dot(sig1_inv)
    
    d_sig0 = np.linalg.det(sigma0)
    d_sig1 = np.linalg.det(sigma1)
    
    L = u_0_s_0 - u_1_s_1
    
    C = (-u_1_s_1.dot(u1) + u_0_s_0.dot(u0))/2 + math.log(phi/(1 - phi)) - (1/2) * (math.log(d_sig1) - math.log(d_sig0))
    
    return (Q/2, L/2, C)

def plotQuadDecisionBoundry(u0, u1, sigma0, sigma1, phy, X, Y):
	q_coeff = calcCoeffQuad(sigma0, sigma1, u0, u1, phy)
	
	Q = q_coeff[0]
	L = q_coeff[1]
	Const_term = q_coeff[2][0][0]
	
	A = Q[0][0]
	B = Q[1][1]
	C = 2 * Q[0][1]
	D = -2 * L[0][0]
	E = -2 * L[0][1]
	# print(Const_term, A, B, C, D, E)
	
	
	x1 = np.linspace(-2, 2, 400)
#	 print(x1)
	alpha = B
	beta = C * x1 + E
	gama = A*(x1**2) + D*x1 + Const_term
	
#	 print(alpha, beta, gama)
	
	
	
	x2 = solveSqrt(alpha, beta, gama)
#	 x2 = [solveSqrt(B, C * i + E, A*(i**2) + D*i + Const_term) for i in x1]
	
	plt.plot(x1, x2)


def solveSqrt(alpha, beta, gama):
#	 print(alpha, beta, gama)
	return (-beta - np.sqrt(beta**2 - 4*alpha*gama)) / (2 * alpha)   


def partA(X, Y):
	u0 = meanFunc(X, Y, "Alaska")
	u1 = meanFunc(X, Y, "Canada")
	sig = covarMatrixFunc(X, Y, u0, u1)
	phi = phiFunc(Y)

	return {"phi": phi, "mean0": u0, "mean1": u1, "covarMatrix": sig}

def partB(X, Y):
	xx0 = getXData(X, Y, val="Alaska")
	x10 = xx0[:,1]
	x20 = xx0[:,2]

	xx1 = getXData(X, Y, val="Canada")
	x11 = xx1[:,1]
	x21 = xx1[:,2]

	plt.title("GDA - Scatter plot of training data")
	plt.xlabel("x1", fontweight="bold")
	plt.ylabel("x2", fontweight="bold")
	plt.scatter(x10, x20,color="r", label="Alaska", marker=".")
	plt.scatter(x11, x21,color="b", label="Canada", marker="x")
	plt.legend(loc="upper right")
	plt.show()

def partC(X, Y, u0, u1, sig, phi):
	# Ax + c = 0
	(A, c) = calcCoeffLinear(u0, u1, sig, phi)

	a1 = A[0][0]
	a2 = A[0][1]
	x1 = X[:, 1].reshape(-1, 1)
	x2 = -(c + a1*x1) / a2
	xx0 = getXData(X, Y, val="Alaska")
	x10 = xx0[:,1]
	x20 = xx0[:,2]

	xx1 = getXData(X, Y, val="Canada")
	x11 = xx1[:,1]
	x21 = xx1[:,2]

	plt.xlabel("x1", fontweight="bold")
	plt.ylabel("x2", fontweight="bold")

	plt.plot(x1, x2, "-", label="Decision boundary")
	plt.scatter(x10, x20,color="r", label="Alaska", marker=".")
	plt.scatter(x11, x21,color="b", label="Canada", marker="x")
	plt.title("GDA - Decision Boundary")
	plt.legend(loc="upper right")
	plt.show()


def partD(X, Y, u0, u1, phi):
	sigma0 = sigmaGeneral(X, Y, u0, u1, 0)
	sigma1 = sigmaGeneral(X, Y, u0, u1, 1)

	print(f"sigma0 = \n {sigma0}")
	print(f"sigma1 = \n{sigma1}")

	q_coeff = calcCoeffQuad(sigma0, sigma1, u0, u1, phi)
	
	Q = q_coeff[0]
	L = q_coeff[1]
	Const_term = q_coeff[2][0][0]

	A = Q[0][0]
	B = Q[1][1]
	C = 2 * Q[0][1]
	D = -2 * L[0][0]
	E = -2 * L[0][1]
	# print(Const_term, A, B, C, D, E)

	x1 = np.linspace(-3, 2, 400)
	alpha = B
	beta = C * x1 + E
	gama = A*(x1**2) + D*x1 + Const_term

	x2 = solveSqrt(alpha, beta, gama)

	plt.title("GDA - Quadratic Decision Boundary")
	plt.plot(x1, x2, label="Decision boundary")

	# plot input point
	xx0 = getXData(X, Y, val="Alaska")
	x10 = xx0[:,1]
	x20 = xx0[:,2]

	xx1 = getXData(X, Y, val="Canada")
	x11 = xx1[:,1]
	x21 = xx1[:,2]

	
	plt.xlabel("x1", fontweight="bold")
	plt.ylabel("x2", fontweight="bold")
	plt.scatter(x10, x20,color="r", label="Alaska", marker=".")
	plt.scatter(x11, x21,color="b", label="Canada", marker="x")
	plt.legend(loc="upper right")
	
	plt.show()




def partD2(X, Y, u0, u1, phi, sig):
	sigma0 = sigmaGeneral(X, Y, u0, u1, 0)
	sigma1 = sigmaGeneral(X, Y, u0, u1, 1)

	q_coeff = calcCoeffQuad(sigma0, sigma1, u0, u1, phi)
	
	Q = q_coeff[0]
	L = q_coeff[1]
	Const_term = q_coeff[2][0][0]

	A = Q[0][0]
	B = Q[1][1]
	C = 2 * Q[0][1]
	D = -2 * L[0][0]
	E = -2 * L[0][1]
	# print(Const_term, A, B, C, D, E)

	x1 = np.linspace(-3, 2, 400)
	alpha = B
	beta = C * x1 + E
	gama = A*(x1**2) + D*x1 + Const_term

	x2 = solveSqrt(alpha, beta, gama)

	plt.title("GDA - Decision Boundary")
	plt.plot(x1, x2, label="Quadratic decision boundary")



	# plot input point
	xx0 = getXData(X, Y, val="Alaska")
	x10 = xx0[:,1]
	x20 = xx0[:,2]

	xx1 = getXData(X, Y, val="Canada")
	x11 = xx1[:,1]
	x21 = xx1[:,2]

	
	plt.xlabel("x1", fontweight="bold")
	plt.ylabel("x2", fontweight="bold")
	plt.scatter(x10, x20,color="r", label="Alaska", marker=".")
	plt.scatter(x11, x21,color="b", label="Canada", marker="x")
	plt.legend(loc="upper right")


	# plot linear boundry
	(A, c) = calcCoeffLinear(u0, u1, sig, phi)

	a1 = A[0][0]
	a2 = A[0][1]
	x1 = X[:, 1].reshape(-1, 1)
	x2 = -(c + a1*x1) / a2

	plt.xlabel("x1", fontweight="bold")
	plt.ylabel("x2", fontweight="bold")

	plt.plot(x1, x2, "-", label="Linear decision boundary")
	
	
	plt.legend(loc="upper right")
	
	plt.show()



	

def main(x_file, y_file):
	data = loadData(x_file, y_file)

	X = data['X']
	Y = data['Y']

	data = partA(X, Y)
	u0 = data["mean0"]
	u1 = data["mean1"]
	sig = data["covarMatrix"]
	phi = data["phi"]

	#---------part(a)------------------
	print("-"*10)
	print("part(a)")
	print("-"*10)
	print(f"phi = {phi}")
	print(f"u0 = \n {u0}")
	print(f"u1 = \n {u1}")
	print(f"co-variance matrix = \n {sig}")
	print("-"*30)
	#-------------------------------------



	#---------part(b)------------------
	print("-"*10)
	print("part(b)")
	print("-"*10)
	partB(X, Y)

	print("-"*30)
	#-------------------------------------

	#---------part(c)------------------
	print("-"*10)
	print("part(c)")
	print("-"*10)
	partC(X, Y, u0, u1, sig, phi)


	print("-"*30)
	#-------------------------------------


	#---------part(d)------------------
	print("-"*10)
	print("part(d)")
	print("-"*10)
	partD(X, Y, u0, u1, phi)


	print("-"*30)
	#-------------------------------------

	#---------part(d2)------------------
	print("-"*10)
	print("part(d2)")
	print("-"*10)
	partD2(X, Y, u0, u1, phi, sig)


	print("-"*30)
	#-------------------------------------










if __name__ == "__main__":
	try:
		# x file, y file 
		x_file = sys.argv[1]
		y_file = sys.argv[2]

		main(x_file, y_file)
	except Exception as e:
		print(e)
		msg = '''
=>python3 q4.py x_file y_file
		'''
		print(msg)

	










