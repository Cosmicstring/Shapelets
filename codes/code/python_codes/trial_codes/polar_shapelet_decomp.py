import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import imsave, imshow, meshgrid
import matplotlib.cm as cm
from scipy import special
from scipy.integrate import quad,dblquad

#-----------------
# berry == Berry et al. MNRAS 2004
# refregier == Massey & Refregier MNRAS 2005i
# nakajima == Nakajima & Bernstein AJ 2007
#----------------

Pi = math.pi
factorial = math.factorial

def coeff(N, M, beta):
    """ 
    Return normalization coefficients for refregier shapelets

    """
    A = ( (-1)**((N - np.abs(M))/2) /
            beta**(np.abs(M) + 1) )
    B = np.sqrt( \
            2*float( factorial( int( (N-np.abs(M))/2 ) ) ) \
            / (float(factorial( (N+np.abs(M))/2 ))))
    C = np.sqrt( 1. / Pi)  
    return A,B, C

def polar_shapelets_refregier(N, M, beta):

    """
    Return callable function for normalized generalized Laguerre polynomials with separated 
    exp(-1j*M*phi) to Cos (M>0) and Sin (M<0) and ordinary radial part for M == 0
    """
    coeff_1, coeff_2, coeff_3 = coeff(N, M, beta)
    gen_laguerre = special.genlaguerre(n=(N-np.abs(M))/2, alpha=np.abs(M))
    
    if (M > 0):
        laguer_N_M = lambda x, phi: coeff_1 * coeff_2 \
                * x**(np.abs(M)) \
                * gen_laguerre(x**2/beta**2) \
                * np.exp(-x**2/2./beta**2) \
                * coeff_3 * np.cos(M*phi) 
    elif (M < 0):
        laguer_N_M = lambda x, phi: coeff_1 * coeff_2 \
                * x**(np.abs(M)) \
                * gen_laguerre(x**2/beta**2) \
                * np.exp(-x**2/2./beta**2) \
                * coeff_3 * np.sin(M*phi)
    elif (M == 0):
        laguer_N_M = lambda x, phi: coeff_1 * coeff_2 \
                * x**(np.abs(M)) \
                * gen_laguerre(x**2/beta**2) \
                * np.exp(-x**2/2./beta**2)
    
    return laguer_N_M



"""
Trial function for testing
BEGIN
"""
def polar_shapelets_radial(N, M, beta):

    coeff_1, coeff_2, coeff_3 = coeff(N, M, beta)
    gen_laguerre = special.genlaguerre(n=(N-np.abs(M))/2, alpha=np.abs(M))

    laguer_N_M = lambda x: coeff_1 *coeff_2 * x**(np.abs(M)) \
            * gen_laguerre(x**2/beta**2) * np.exp(-x**2/(2*beta**2))
    return laguer_N_M

def polar_shapelets_angular(N, M, beta):
    
    coeff_1, coeff_2,coeff_3 = coeff(N,M,beta)

    laguer_N_M = lambda x: 1

    if (M>0):
        laguer_N_M = lambda x: coeff_3 * np.cos(M*x)
    elif (M<0):
        laguer_N_M = lambda x: coeff_3 * np.sin(M*x)
        
    return laguer_N_M

def coeff_nakajima(p,q):
    
    """ Calculate coefficients from nakajima papre
    
    @param p,q integers by which indexation of wave functions is made
    """
    A = (-1)**q / (2*Pi)
    B = np.sqrt( float(factorial(q)) / float(factorial(p)) )
    C = .5*np.sqrt((p-q) / Pi)
    
    return A, B, C

def polar_shapelets_nakajima(p,q,beta):

    """ Return callable shapelet function defined by nakajima eq 27

    @param p,q integers for indexation of wave functions
    @param beta scale control factor
    """
    coeff_1, coeff_2, coeff_3 = coeff_nakajima(p,q)
    m = p-q
    
    gen_laguerre = special.genlaguerre(n=q,alpha = m)
    
    psi_p_q = lambda r,phi: coeff_1 * coeff_2 \
            * (r/beta)**m \
            * np.exp(-r**2 / (2 * beta**2)) \
            * gen_laguerre(r**2 / beta**2) \
            * coeff_3 * np.cos(m*phi)
    return psi_p_q

"""
Trial functions for testing
END
"""

#From Berry et al. Eq (12)

def coeff_berry(N,M,beta):
    return (2.*float(factorial((N-np.abs(M)/2)))/(beta**2*float(factorial((N+np.abs(M))/2))))**(0.5)

def polar_shapelets_berry(N,M, beta):

    coeff = coeff_berry(N,M,beta)
    gen_laguerre = special.genlaguerre(n = (N-np.abs(M))/2, alpha = np.abs(M))

    laguer_N_M = lambda r, phi: coeff * (r/beta)**(np.abs(M)) * np.exp(-(r**2/beta**2)/2.) * gen_laguerre(r**2/beta**2) * np.exp(-1j*M*phi)
    
    return laguer_N_M


def plot_shapelets_test_image(N1,M1,N2,M2,N3,M3, beta):

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
    
    a = 0
    b = 0
    c = 1

    func = a*polar_shapelets_refregier(N1,M1,beta)(R,phi) + b*polar_shapelets_refregier(N2, M2, beta)(R, phi) + c*polar_shapelets_refregier(N3,M3,beta)(R,phi)
    im = imshow(func, cmap=cm.bwr)
    
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
    plt.show()

if __name__ == "__main__":
    
  
    check = -1; beta = 1.; check_dep = 1;

    #plot_shapelets_test_image(0,0, 1,1, 1,-1, 1.2)
    

    if (check == 1):
        for n in xrange(10):
            for m in xrange(-n,n+1,2):
                print(n,m)
                #print('berry')
                #print(dblquad(lambda r,phi: r*polar_shapelets_berry(n,m,beta)(r,phi).conjugate()*polar_shapelets_berry(n,m,beta)(r,phi), 0, np.inf, lambda r: 0, lambda r: 2*Pi))
                print('refregier')
                print(quad(lambda x: polar_shapelets_angular(n,m,beta)(x)**2, 0,2*Pi)) 
    #print(polar_shapelets(1,1,0,Pi/2, 1.))
    
    ## Test linear dependance
    if check_dep == 1:
        X = np.linspace(-5, 5, 78)
        Y = np.linspace(-5, 5, 78)
        Xv, Yv = meshgrid(X,Y)
        R = np.sqrt(Xv**2 +  Yv**2)
        phi = np.zeros_like(R)
        for i in xrange(np.shape(Xv)[0]):
            for j in xrange(np.shape(Xv)[1]):
                phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])
        for n1 in xrange(10):
            for m1 in xrange(-n1,n1+1,2):
                for n2 in xrange(10):
                    for m2 in xrange(-n2,n2+1,2):
                        dot_prod = np.dot(polar_shapelets_refregier(n1,m1,beta)(R,phi).flatten(), polar_shapelets_refregier(n2,m2,beta)(R,phi).flatten())
                        if dot_prod > 0.01:
                            print n1,m1, ' dot ', n2,m2, ':'
                            print dot_prod
