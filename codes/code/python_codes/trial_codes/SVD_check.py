import numpy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
from polar_shapelet_decomp import polar_shapelets_refregier as p_shapelet_refr
#import sys

#sys.path.append('/home/kostic/Documents/codes/code/')

import shapelet_dicts as sh_dict
import pdb;pdb.set_trace()

k = 0
n = 1.24
N_data = 100


x = 10*np.random.rand(N_data)
y = k*x+ n

sigma_x = 10*np.random.rand(N_data)
sigma_y = 2*np.random.rand(N_data)

x_data = np.zeros_like(x)
y_data = np.zeros_like(y)
A = np.zeros((N_data,2))
z = 1.

for i in xrange(len(x)):
    x_data[i] = x[i] + z*sigma_x[i]
    y_data[i] = y[i] + z*sigma_y[i]
    z *= -1.

A[:,0] = x_data
A[:,1] = np.ones(len(x_data))

U, s, VT = linalg.svd(A, full_matrices = True, compute_uv = True)

V = VT.transpose()
S_dual = np.eye(N_data,2)*(1./s)

coeffs = np.dot(V, np.dot(S_dual.transpose(),np.dot(U.transpose(), y_data)))

coeffs_lstsq = linalg.lstsq(A,y_data)[0]

n_max = 20
beta = 1.
coeff_y = np.zeros(n_max)
D = np.zeros((len(x),n_max))

for n in xrange(n_max):
    basis_vec = sh_dict.shapelet1d(n,s=beta)(x)
    D[:,n] = basis_vec
    coeff_y[n] = np.dot(basis_vec, y)

print(np.dot(D, coeff_y))
print(y)


print 'Comparing 1D performance of SVD and Lstsq'

# Test the SVD fit of shapelet decomp
U, s, VT = linalg.svd(D, full_matrices = True, compute_uv = True)
V = VT.transpose()

dual_s = 1./s
S = np.zeros_like(D)
S_dual = np.zeros_like(D)

for i in xrange(len(dual_s)):
    S[i,i] = s[i]
    S_dual[i,i] = dual_s[i]

print 'SVD decomp'
print(np.allclose(np.dot(U, np.dot(S, V.transpose())), D))

coeffs_SVD = np.dot(V, np.dot(S_dual.transpose(),np.dot(U.transpose(),y)))
print coeffs_SVD

y_SVD = np.dot(D, coeffs_SVD)

print(y_SVD)
print(y)

# Test the lstsq fit of the shapelet decomp
print('Lstsq decomp')
coeffs_Lstsq = linalg.lstsq(D,y)[0]
y_Lstsq = np.dot(D, coeffs_Lstsq)


print 'Compare sigma'
print np.var(y_Lstsq)/np.var(y_SVD)

#plt.plot(x,y, 'bo', x, y_Lstsq, 'r--')
#plt.show()

print 'Comparing 2D performance of SVD and Lstsq'

# Make 2D shapelet images for testing the coeffs given by SVD and Lstsq

from pylab import imread,imshow
import matplotlib.image as mpimg
import math

n_max = 10
beta_polar = 1.

test_image_matrix = np.loadtxt("test_image_matrix.txt", unpack = True)
print(test_image_matrix.dtype)
test_image = mpimg.imread(fname = 'test_image.png')

size_X = test_image.shape[0]
size_Y = test_image.shape[1]

test_image = test_image_matrix.flatten()

X = np.linspace(-5,5, size_X)
Y = np.linspace(-5,5, size_Y)
Xv,Yv = np.meshgrid(X,Y)

# Initialize the meshgrid for polar basis shapelets
R = np.sqrt(Xv**2 + Yv**2)
phi = np.zeros_like(R)
for i in xrange(Xv.shape[0]):
    for j in xrange(Xv.shape[1]):
        phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])

basis_vec_polar = np.zeros_like(test_image)
reconst = np.zeros_like(test_image)

coeff_polar = np.zeros(2*n_max**2 + n_max + 10)
D_im = np.zeros((len(basis_vec_polar), len(coeff_polar)))

k = 0

# Make the coefs and basis matrix

for n in xrange(n_max):
    for m in xrange(-n,n+1,2):
        basis_vec_polar = p_shapelet_refr(n,m,beta_polar)(R,phi).flatten()
        D_im[:,k] = basis_vec_polar
        coeff_polar[k] = np.dot(basis_vec_polar, test_image)
        k+=1

coeff_polar = np.asarray(coeff_polar)

# SVD for 2D image of template shapelets
U, s, VT = linalg.svd(D_im, full_matrices = True, compute_uv = True)

V = VT.transpose()

dual_s = 1./s
S = np.zeros_like(D_im)
S_dual = np.zeros_like(D_im)

for i in xrange(len(dual_s)):
    S[i,i] = s[i]
    S_dual[i,i] = dual_s[i]

print test_image.shape
print V.shape, S.shape, S_dual.shape, U.shape

coeffs_SVD_polar = np.dot(V, np.dot(S_dual.transpose(),np.dot(U.transpose(),test_image)))

print 'nonzero_coefs_SVD', np.count_nonzero(coeffs_SVD_polar)
print coeffs_SVD_polar

#imshow(np.dot(D_im,coeffs_SVD_polar).reshape(size_X,size_Y))
#plt.show()
#plt.clf()

# Lstq for 2D image of template shapelets
coeffs_Lstsq_polar = linalg.lstsq(D_im,test_image)[0]

print 'nonzero_coefs_Lstsq', np.count_nonzero(coeffs_Lstsq_polar)
print coeffs_Lstsq_polar
#imshow(np.dot(D_im,coeffs_Lstsq_polar).reshape(size_X,size_Y))
#plt.show()
