#!/usr/bin/env python3

"""
	QR Factorization (nxn)
"""

__filename__ = "QRfact.py"
__author__ = "L.J. Brown"

# external
import numpy as np

# my lib
from function_performance_timer import time_function

def rand_mat(m,n,r=[-1,1]):
    M = (r[1]-r[0])*np.random.rand(m,n) - (r[1])*np.ones((m,n))
    return M

def random_matrix(shape=(1,1), max_value=1, min_value=0):
	""" 
		Return random matrix.
		:param shape: integer tuple of size two containing dimensions of Matrix. ex: shape=(n,m).
		:param max_value: Maximum value an element of the generated matrix can take.
		:param min_value: Minimum value an element of the generated matrix can take.
		:returns: Randomly generated matrix.
	"""
	n, m = shape
	R = np.random.rand(n,m)*np.random.randint(low=min_value, high=max_value)
	return R

def random_vector(n, min_value=0, max_value=1):
	""" 
		Return random matrix.
		:param n: integer length containing number of elements for the generated vector.
		:param max_value: Maximum value an element of the generated vector can take.
		:param min_value: Minimum value an element of the generated vector can take.
		:returns: Randomly generated vector.
	"""
	x = np.random.rand(n)*np.random.randint(low=min_value, high=max_value)
	return x

def bsub_row(U,y):
	for i in range(U.shape[0]-1,-1,-1): 
		for j in range(i+1, U.shape[1]):
			y[i] -= U[i,j]*y[j]
		y[i] = y[i]/U[i,i]
	return y

def update_QR(Qt,R,a,b):
	m = 1/np.sqrt(R[b,b]**2 + R[a,b]**2)
	c, s = R[b,b]*m, R[a,b]*m

	ra = R[a,:].copy()
	rb = R[b,:].copy()

	R[a,:] = -s*rb +c*ra
	R[b,:] = c*rb +s*ra

	qa = Qt[a,:].copy()
	qb = Qt[b,:].copy()

	Qt[a,:] = -s*qb +c*qa
	Qt[b,:] = c*qb +s*qa

def QRfact(A):
	n = A.shape[0]
	R = A
	Qt = np.eye(n)

	for j in range(n):
		for i in range(j+1,n):
			update_QR(Qt,R,i,j)

	return [Qt.T,R]

def QRsolve(Q,R,b):
	# 1). c = Q.T @ b
	# 2). solve Rx = c (bsub)

	c = Q.T @ b
	x = bsub_row(R,c)

	return x
