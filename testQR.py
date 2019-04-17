#!/usr/bin/env python3

"""
	Test QR Decomposition Function PErformance
"""

__filename__ = "testQR.py"
__author__ = "L.J. Brown"

# external
import numpy as np

# my lib
from function_performance_timer import time_function
from QRfact import QRfact, QRsolve


#
# Test Settings
#

NUM_SIZES = 100
NUM_TRAILS = 1
MIN_DIM_NXN = 1e2
MAX_DIM_NXN = 1e3
MIN_VAL = -0.5e2
MAX_VAL = 0.5e2

#
# Helper Methods
#

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


#
# Run Tests 
#

if __name__ == "__main__":

	sizes = np.linspace(MIN_DIM_NXN, MAX_DIM_NXN, num=NUM_SIZES).astype(int)
	decomp_times = np.zeros(shape=(NUM_TRAILS*NUM_SIZES,))

	# log to screen
	print("\n\nTesting square QR decomposition on sizes: %s\n" % (sizes))

	for i, n in enumerate(sizes):

		# Create random nxn matrix
		A = random_matrix(shape=(n,n), min_value=MIN_VAL, max_value=MAX_VAL) 

		# create random solution vector
		x = random_vector(n, min_value=MIN_VAL, max_value=MAX_VAL)

		# calculate b
		b = A @ x

		# In
		I = np.eye(n)

		# Perform QR decomposition and time method
		decomposition_time, [Q, R] = time_function(QRfact, np.copy(A), return_function_output=True)
		solve_time, x = time_function(QRsolve, Q, R, b, return_function_output=True)

		# compute error
		decomposition_error = np.linalg.norm( A - Q @ R )
		orthogonal_columns_error = np.linalg.norm( I - Q.T @ Q )
		orthogonal_rows_error = np.linalg.norm( I - Q @ Q.T )
		residual_norm = np.linalg.norm( b - A @ x )

		# record stats
		decomp_times[i] = decomposition_time

		#
		# Display Error
		#

		# display runtime and relative error of both methods for size n
		print("\n\n")
		print("\tSize (%sx%s):" % (n,n))
		print("\tDecomposition time (sec): %s" % (decomposition_time*1e-3))
		print("\nSolve time (sec): %s" % (solve_time*1e-3))
		print("\t||A - QR||2 = ", decomposition_error)
		print("\t||b - Ax||2 = ", residual_norm)
		print("\t||I - QTQ||2 = ", orthogonal_columns_error)
		print("\t||I - QQT||2 = ", orthogonal_rows_error)
		print("\n\n")


	import matplotlib.pyplot as plt
	plt.plot(sizes, decomp_times, 'b-', label='data')
	plt.xlabel('n')
	plt.ylabel('Decomp time (sec)')
	plt.show()
