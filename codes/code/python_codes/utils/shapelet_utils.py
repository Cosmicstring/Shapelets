import numpy as np
import math
from scipy import special
from scipy.special import hermitenorm

"""
------------------
berry == Berry et al. MNRAS 2004
bosch == Bosch J. AJ 2010 vol 140 pp 870 - 879
nakajima == Nakajima R., Bernstein G., APJ 2007, vol 133, pp 1763-1779
refregier == Refregier MNRAS 2003 / Shapelets I and II
------------------
"""


factorial = math.factorial
Pi = np.pi

def get_max_order(basis, n_max):
    """
    Calculating max order of shapelets included in order to get
    a symmetric triangle. This will return the first smaller one
    than n_max which provides a complete triangle
    
    Deprecated:
    -----------
    basis : Basis of the decomp.
    // It doesn't matter if it's polar or XY the coeffs triangle is the same
    
    Parameters:
    -----------
    n_max : Chosen max number by user

    ---------------
    For example

    n_max = 5

    XY
    return 1 -----> Combinations of shapelets available (0,0), (0,1), (1,0)
    // If 2 is included the number of combinations would exceed n_max and 
    // triangle won't be symmetric
    return 1 -----> Combinations of shapelets (0,0), (-1,1), (1,1)
    // Same as above
    """

    #xy_max_order = lambda n: int(np.floor(np.sqrt(n) - 1))
    #polar_max_order = lambda n: int(np.floor( (-3 + np.sqrt(9 + 8*(n-1)))/2.))
    
    ## Get the symmetric triangle for XY and same for Polar
    ## It's the same because for XY I look at the m+n, and for Polar just n
    max_order = lambda n: int(np.floor( (-3 + np.sqrt(9 + 8*(n-1)))/2.))

    return max_order(n_max)

def sum_max_order(basis, n):
    """
    Given the maximum order of shapelet included
    calculate how many combinations of shapelets there are
    
    -------------
    For example:
    n = 1
    
    XY basis / Elliptical basis
    return 3 ---> Combinations (0,0), (0,1), (1,0) 
    // Because in the decomp I hace conditon n+m<= 1, in that way i get symmetric triangle
    Polar
    return 3 ---> Combinations (0,0), (-1,1), (1,1) 
    // If 2 would be included then the order of shapelet would exceed 1 
    """
    return n*(n+1)/2 + n + 1

##Define orthonormal basis - shapelets
def shapelet1d(n,x0=0,s=1):

    """Make a 1D shapelet template to be used in the construction of 2D shapelets

    n 	: Integer representing energy quantum number
    x0 	: Integer defining the centroid
    s 	: Float which is the same as beta parameter in refregier

    """
    def sqfac(k):
        fac = 1.
        for i in xrange(k):
            fac *= np.sqrt(i+1)
        return fac

    u = lambda x: (x-x0)/s
    fn = lambda x: (1./(2*np.pi)**0.25)*(1./sqfac(n))*hermitenorm(n)(u(x))*np.exp(-0.25*u(x)**2) 
    return fn

def shapelet2d(m,n,x0=0,y0=0,sx=1,sy=1):
    
    """Make a 2D shapelet template function to be used in image decomposition later on
    
    n,m  	: Integers representing energy_x and energy_y quantum numbers, respectively
    x0,y0 	: Integers defining image centroid - X and Y coordinate, resepectively
    sx,sy 	: Floats setting the beta scale for the X and Y shapelet space, respectively
    
    """

    u = lambda x: (x-x0)/sx
    v = lambda y: (y-y0)/sy
    fn = lambda x,y: np.outer(shapelet1d(m)(u(x)),shapelet1d(n)(v(y)))
    return fn

def elliptical_shapelet(m,n,x0=0,y0=0,sx=1,sy=1,theta=0):
    
    """ Make elliptical shapelets to be used in the decomposition of images later on

    n,m  	: Integers representing energy_x and energy_y quantum numbers, respectively
    x0,y0 	: Integers defining image centroid - X and Y coordinate, resepectively
    sx,sy 	: Floats setting the beta scale for the X and Y shapelet space, respectively
    theta 	: Float for setting up the ellipse orientation
    
    """
    u = lambda x,y: (x-x0)*np.cos(theta)/sx + (y-y0)*np.sin(theta)/sy
    v = lambda x,y: (y-y0)*np.cos(theta)/sy - (x-x0)*np.sin(theta)/sx
    fn = lambda x,y: shapelet1d(m)(u(x,y)) * shapelet1d(n)(v(x,y))
    return fn

def coeff(N, M, beta):
    """ 
    Return nomalizing coefficients in front of the
	shapelet functions - refering to refregier paper 

    """
    A = ( (-1)**((N - np.abs(M))/2) /
            beta**(np.abs(M) + 1) )
    B = np.sqrt( \
            2*float( factorial( int( (N-np.abs(M))/2 ) ) ) \
            / (float(factorial( (N+np.abs(M))/2 ))))
    C = np.sqrt( 1. / Pi)  
    return A,B, C

def polar_shapelets_radial(N, M, beta):
    """
    Return a callable function of the radial part
    of the shapelets as in @refregier. Essentially
    these are normalized Laguer polynomials
    """
    coeff_1, coeff_2, coeff_3 = coeff(N, M, beta)
    gen_laguerre = special.genlaguerre(n=(N-np.abs(M))/2, alpha=np.abs(M))

    laguer_N_M = lambda x: coeff_1 *coeff_2 * x**(np.abs(M)) \
            * gen_laguerre((x/beta)**2) * np.exp(-0.5*(x/beta)**2)
    return laguer_N_M

def polar_shapelets_angular(N, M, beta):
    """
    Return callable function of the angular part 
    of the shapelets (modified from paper @refregier)
    """
    
    coeff_1, coeff_2,coeff_3 = coeff(N,M,beta)

    laguer_N_M = lambda x: 1

    if (M>0):
        laguer_N_M = lambda x: coeff_3 * np.cos(M*x)
    elif (M<0):
        laguer_N_M = lambda x: coeff_3 * np.sin(M*x)
        
    return laguer_N_M

def polar_shapelets_refregier(N, M, beta, theta = 0.):

    """
    Return callable function for normalized generalized Laguerre polynomials with separated 
    exp(-1j*M*phi) to Cos (M>0) and Sin (M<0) and ordinary radial part for M == 0
    """
    coeff_1, coeff_2, coeff_3 = coeff(N, M, beta)
    #gen_laguerre = special.genlaguerre(n=(N-np.abs(M))/2, alpha=np.abs(M)) 

    polar_N_M = lambda x,phi: polar_shapelets_radial(N,M,beta)(x) \
            * polar_shapelets_angular(N,M,beta)(phi + theta)

    return polar_N_M

def decompose_cartesian(basis,\
        D, base_coefs, \
        shapelet_reconst, signal, make_labels, \
        label_arr,\
        n_max,N1,N2,\
        x0,y0,sigma,\
        X,Y,\
        q=1., theta = 0.):
    """
    Decompose into XY or XY_Elliptical basis, return obtained reconstruction with shapelets
    and also construct label_array and basis matrix/dictionary D, return the inner product reconst.
	
	Input parameters:
	-----------------
	
	basis				:	String representing the basis of dictionary
	D					:	Matrix of dim(N1xN2xlen(beta_array)), representing the dictionary. Initialized to all zeros at the beginning
	base_coefs			:	1D array of floats for storing inner product coefficients. Initialized to zero at the beginning
	shapelet_reconst	:	1D array of floats for storing the obtained inner product reconstruciton of the image
	signal				:	1D array of pixel values of the initial image, which is to be reconstructed
	make_labels			:	Boolean variable controling the manipulation with label_arr
	label_arr			:	1D array of char for storing the shapelet labels which go into D. Initialized to empty strings.
	n_max, N1, N2		:	Integers defining the maximum number of shapelets in the given beta scale (sigma), 
							upper limit on energy_x and energy_y quantum numbers respectively
	x0,y0				:	Integers defining the centroid of the image
	sigma				:	Float which represents the beta scale of the shapelets. Obtained from the FindAdaptiveMom()
	X,Y					:	1D arrays which are used to make the meshgrid for sampling the 2D shapelets
	q,theta				:	Floats defining ellipticity and orientation of shapelets
	
	Returns:
	--------
	
	shapelet_reconst 	:	1D array of inner product image reconstruction
	
    """
    
    a = sigma / np.sqrt(q)
    b = sigma * np.sqrt(q)

    ## Make a meshgrid for sampling the function
    Xv,Yv = np.meshgrid(X,Y)

    ## Add one more diagonal to the basis matrix
    max_order = get_max_order(basis,n_max) + 1
    for k in xrange(N1*N2):
        m,n = k/N1, k%N1
        if (m+n <= max_order): 
            if make_labels:
                ## To be consistent with indexation of the basis matrix
                ## D
                label_arr[k] = (str("(%d, %d)" % (n,m)))
            if basis == 'XY':
                arr = shapelet2d(m,n,x0=x0,y0=y0,sx=sigma,sy=sigma)(X,Y).flatten() 
            elif basis == 'XY_Elliptical':
                arr = elliptical_shapelet(m,n,x0=x0,y0=y0,sx=a,sy=b,theta=theta)(Xv,Yv).flatten()
            D[:, k] = arr
            arr_norm2 = np.dot(arr,arr)
            coef = np.dot(arr,signal)
            if(coef==0):
                base_coefs[n,m] =0
            else:
                base_coefs[n,m] = coef/np.sqrt(arr_norm2)
                shapelet_reconst += (coef*arr)/arr_norm2

    return shapelet_reconst

def decompose_polar(basis,\
       D,base_coefs,\
       shapelet_reconst, signal, make_labels, \
       label_arr,\
       n_max, N1,N2,\
       x0,y0,sigma,\
       X,Y,\
       polar_basis = 'refregier',\
       q=1., theta = 0.):

    """
    Decompose into Polar or Polar_Elliptical shapelet basis, return shapelet reconstruction
    and constract basis matrix as well as label_array, return the inner product
    reconstruciton
	
	Input parameters:
	-----------------
	
	basis				:	String representing the basis of dictionary
	D					:	Matrix of dim(N1xN2xlen(beta_array)), representing the dictionary. Initialized to all zeros at the beginning
	base_coefs			:	1D array of floats for storing inner product coefficients. Initialized to zero at the beginning
	shapelet_reconst	:	1D array of floats for storing the obtained inner product reconstruciton of the image
	signal				:	1D array of pixel values of the initial image, which is to be reconstructed
	make_labels			:	Boolean variable controling the manipulation with label_arr
	label_arr			:	1D array of char for storing the shapelet labels which go into D. Initialized to empty strings.
	n_max, N1, N2		:	Integers defining the maximum number of shapelets in the given beta scale (sigma), 
							upper limit on energy_x and energy_y quantum numbers respectively
	x0,y0				:	Integers defining the centroid of the image
	sigma				:	Float which represents the beta scale of the shapelets. Obtained from the FindAdaptiveMom()
	X,Y					:	1D arrays which are used to make the meshgrid for sampling the 2D shapelets
	polar_basis			: 	String which chooses the type of polar basis to use 
							// This was used to check the difference between the nakajima paper and refregier paper equations
							// it seems that the refregier shapelets are normalized properly
	q,theta				:	Floats defining ellipticity and orientation of shapelets
	
	Returns:
	--------
	
	shapelet_reconst 	:	1D array of inner product image reconstruction
    """
    
    Xv, Yv = np.meshgrid((X-x0),(Y-y0))     
    Phi = np.arctan2(Yv,Xv)
    R = np.sqrt(Xv**2 + Yv**2) 
    
    ## If basis is Polar_Elliptical edit the meshgrid
    ## so that surface area is perseved but the circle is stretched
    if basis == 'Polar_Elliptical':
       R = R*np.sqrt(q * np.cos(Phi+theta)**2 + np.sin(Phi+theta)**2 / q) 

    ## Set the counter for columns of basis matrix D

    k_p = 0

    ## For symmetric coeffs triangle solve
    ## n^2 + 2n + 2 = 2n_max, positive solution, such that
    ## Sum(i+1)_{i in [0,n]} <= n_max

    ## Add the one order higher polar
    max_order = get_max_order(basis,n_max) + 1
    
    ## This needs to go to max_order + 1, because maximum order would not be included
    ## in the iteration
    for n in xrange(max_order + 1):
        for m in xrange(-n,n+1,2):
            
            ## n_max - defined as:
            ## theta_max (galaxy size) / theta_min (smallest variation size) - 1  

            if make_labels:
                ## To be consistent with the indexation of the basis matrix
                ## D
                label_arr[k_p] = (str("(%d, %d)" % (n,m))) 
            
            arr_res = \
                    polar_shapelets_refregier(n,m,sigma,theta = theta)(R,Phi).flatten()
            ## Make the basis matrix D
            D[:,k_p] = arr_res 
            ## Calculate the norms of basis vectors and coefficients
            arr_norm2_res = np.dot(arr_res,arr_res)
            coef_res= np.dot(arr_res, signal)

            ## Add coefficients to basis_coefs array for later use
            ## Make the shapelet reconstruction
            if (coef_res==0): 
                base_coefs[k_p] = 0 
            else: 
                base_coefs[k_p] = coef_res/np.sqrt(arr_norm2_res)
                shapelet_reconst += coef_res*arr_res/arr_norm2_res
            k_p += 1
    
    return shapelet_reconst

def decompose_compound(basis,\
       D,base_coefs,beta_array, \
       shapelet_reconst, signal, make_labels, \
       label_arr,\
       n_max, N1,N2,\
       x0,y0,\
       X,Y,\
       polar_basis = 'refregier',\
       q=1., theta = 0.):
    
    """
    Decompose into the compound basis / Polar_Elliptical or XY_Elliptical / (@cite bosch) with provided basis matrix and signal
    return the obtained inner product reconstruction
	
	Input parameters:
	-----------------
	
	basis				:	String representing the basis of dictionary
	D					:	Matrix of dim(N1xN2xlen(beta_array)), representing the dictionary. Initialized to all zeros at the beginning
	base_coefs			:	2D array of floats dim(len(beta_array) x (N1*N2))for storing inner product coefficients. Initialized to zero at the beginning
	beta_array			:	1D array of floats which contains chosen beta scales for the compound basis
	shapelet_reconst	:	2D array of floats dim(len(beta_array) x (signal.shape[0]*signal.shape[1])) \
							for storing the obtained inner product reconstruciton of the image
	signal				:	1D array of pixel values of the initial image, which is to be reconstructed
	make_labels			:	Boolean variable controling the manipulation with label_arr
	label_arr			:	1D array of char for storing the shapelet labels which go into D. Initialized to empty strings.
	n_max, N1, N2		:	Integers defining the maximum number of shapelets in the given beta scale (sigma), \
							upper limit on energy_x and energy_y quantum numbers respectively
	x0,y0				:	Integers defining the centroid of the image
	sigma				:	Float which represents the beta scale of the shapelets. Obtained from the FindAdaptiveMom()
	X,Y					:	1D arrays which are used to make the meshgrid for sampling the 2D shapelets
	polar_basis			: 	String which chooses the type of polar basis to use \
							// This was used to check the difference between the nakajima paper and refregier paper equations
							// it seems that the refregier shapelets are normalized properly
	q,theta				:	Floats defining ellipticity and orientation of shapelets
	
	Returns:
	--------
	
	shapelet_reconst 	:	2D array of dime(len(beta_array) x (signal.shape[0]*signal.shape[1])) inner product image reconstruction
    """


    ## Just put different betas (range TBD) basis into a big basis matrix 
    ## let the algorithms pick the weights. Watch out that the order stays symmetric 
    step = 0
    max_order = get_max_order(basis, n_max)
    
    ## Define the meshgrid for the sampling of
    ## elliptical shapelets 
    
    Xv,Yv = np.meshgrid(X,Y)
    
    Xv_polar, Yv_polar = np.meshgrid(X-x0, Y-y0)
    Phi = np.arctan2(Yv_polar,Xv_polar)
    R = np.sqrt(Xv_polar**2 + Yv_polar**2) 
    ## If basis is Compound_Polar edit the meshgrid
    ## so that surface area is perseved but the circle is stretched
    R = R*np.sqrt(q * np.cos(Phi+theta)**2 + np.sin(Phi+theta)**2 / q) 
    
    for sigma in beta_array:
        ## Reset indexation for a new basis
        ## with step the basis block is controled
        ##      |vec_basis_1_1   ... / vec_basis_2_1   .. /     |
        ##      |vec_basis_1_2   ... / vec_basis_2_2   .. /     |
        ## D=   .   .    ...         .  ..             .. .  ...|
        ##      |vec_basis_1_N-1 ... / vec_basis_2_N-1 .. /     |
        ##      |vec_basis_1_N   ... / vec_basis_2_N   .. /     |
        ## 
        ## vec_basis_i_j is actually arr variable which is the basis
        ## vector of ith (i is in len(beta_array), 
        ## corresponds to the number of beta values) basis 
        ## and j is it's jth coordinate value
        a = sigma / np.sqrt(q)
        b = sigma * np.sqrt(q)
        if 'XY' in basis:
            for k in xrange(N1*N2):
                ## Number of cols of D is N1*N2*len(beta_array)
                ## beta should change when one whole basis is sweeped by k
                m,n = k/N1, k%N1
                if (m+n <= max_order): 
                    if make_labels:
                        label_arr[k+(N1*N2)*step] = (str("(%d, %d)" % (n,m)))
                    if 'XY' in basis:
                        arr = elliptical_shapelet(m,n,x0,y0,sx=a,sy=b,theta=theta)(Xv,Yv).flatten()
                    D[:,k+(N1*N2)*step] = arr
                    arr_norm2 = np.dot(arr,arr)
                    coef = np.dot(arr, signal)
                    if coef == 0:
                        ## Watch not to overwrite the existing 
                        base_coefs[n, m + N2*step] = 0
                    else:
                        base_coefs[n, m + N2*step] = coef / np.sqrt(arr_norm2)
                        shapelet_reconst[step] += coef*arr/arr_norm2
        elif 'Polar' in basis:
            
            k_p = 0
            for n in xrange(max_order + 1):
                for m in xrange(-n,n+1,2): 
                    ## n_max - defined as:
                    ## theta_max (galaxy size) / theta_min (smallest variation size) - 1  

                    if make_labels:
                        ## To be consistent with the indexation of the basis matrix
                        ## D
                        label_arr[k_p+(N1*N2)*step] = (str("(%d, %d)" % (n,m))) 
                    arr_res = \
                            polar_shapelets_refregier(n,m,sigma,theta = theta)(R,Phi).flatten()
                    ## Make the basis matrix D
                    D[:,k_p + (N1*N2)*step] = arr_res 
                    ## Calculate the norms of basis vectors and coefficients
                    arr_norm2_res = np.dot(arr_res,arr_res)
                    coef_res= np.dot(arr_res, signal)

                    ## Add coefficients to basis_coefs array for later use
                    ## Make the shapelet reconstruction
                    if (coef_res==0): 
                        base_coefs[k_p+(N1*N2)*step] = 0 
                    else: 
                        base_coefs[k_p+(N1*N2)*step] = coef_res/np.sqrt(arr_norm2_res)
                        shapelet_reconst[step] += coef_res*arr_res/arr_norm2_res
                    k_p += 1

        ## Basis finished increase the step
        step += 1

    return shapelet_reconst
