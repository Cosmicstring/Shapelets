import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import imsave, imshow, meshgrid
import matplotlib.cm as cm
from scipy import special
from scipy.integrate import quad,dblquad

import polar_shapelet_decomp as p_shapelet
import sys

sys.path.append("/home/kostic/Documents/codes/code/python_codes/")

import shapelet_dicts as sh_dict


def elliptical_shapelets_test_image(N1,M1,beta,q):

    """
    Make a 2D grid for visualizing the shapelets
    """
    X = np.linspace(-5, 5, 70)
    Y = np.linspace(-5, 5, 70)
    Xv, Yv = meshgrid(X,Y)
    R = np.sqrt(Xv**2 +  Yv**2)
    phi = np.zeros_like(R)
    for i in xrange(np.shape(Xv)[0]):
        for j in xrange(np.shape(Xv)[1]):
            phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])
    
    a = beta / np.sqrt(q)
    b = beta * np.sqrt(q)

    func = sh_dict.elliptical_shapelet(N1,M1,sx=a,sy=b)(X, Y)
    im = imshow(func, cmap=cm.bwr)
    
    """
    # Save the Ns and Ms of shapelets for the test image, and the appropriate coeffs
    f = open('test_coeffs.txt','w')
    f.write("a=%.3f\nb=%.3f\nc=%.3f\n \
            First shapelet:\tN=%d\tM=%d\n \
            Second shapelet:\tN=%d\tM=%d\n \
            Third shapelet:\tN=%d\tM=%d\n" % (a,b,c,N1,M1,N2,M2,N3,M3))
    f.close()
    # Save the image matrix so that there is no loss in resolution
    f1 = open('test_image_matrix.txt', 'w')
    
    print 'func shape: ', func.shape
    
    for i in xrange(func.shape[0]):
        for j in xrange(func.shape[1]):
            f1.write("%.6f\t" % (func[i,j]))
    f1.close()
    
    # Save the image
    imsave(fname='test_image.png', arr = func, cmap=cm.bwr)
    """
    plt.show()

if "__main__" == __name__:

    elliptical_shapelets_test_image(3,3,1.2,2.)
    
