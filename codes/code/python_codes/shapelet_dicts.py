"""
------------------
berry == Berry et al. MNRAS 2004
refregier == Refregier MNRAS 2003 / Shapelets I and II
bosch == Bosch J. AJ 2010 vol 140 pp 870 - 879
------------------
"""

import sys,os
import numpy as np
from scipy.special import hermitenorm
from scipy.integrate import quad


from utils.shapelet_utils import *
from utils.galsim_utils import *

#import pdb; pdb.set_trace()

## About the warning of the a lot of images
#import matplotlib
#matplotlib.use("Agg")

import pyfits
import galsim
import math


## Custom solver functions
from solver_routines import *
from plotting_routines import *

DEBUG = 0

def check_orthonormality():
    M,N = 3, 3
    X = np.arange(-16,16,0.01)
    Y = np.arange(-16,16,0.01)

    for k1 in xrange(M*N):
        m1,n1 = k1/N, k1%N
        b1 = shapelet2d(m1,n1,x0=0.3,y0=0.4,sx=2,sy=3)(X,Y)
        for k2 in xrange(M*N):
            m2,n2 = k2/N, k1%N
            b2 = shapelet2d(m2,n2,x0=0.3,y0=0.4,sx=2,sy=3)(X,Y)

            print m1-m2,n1-n2,np.sum(b1*b2)*(0.01/2)*(0.01/3)

def calculate_spark(D):
    pass

def show_some_shapelets(M=5,N=5, theta = 0., sigma=1., q=1., basis = 'Polar_Elliptical'):
    
    X0 = np.linspace(-8,8,17)
    Y0 = np.linspace(-8,8,17)
    
    X = X0*np.cos(theta) + Y0*np.sin(theta)
    Y = -X0*np.sin(theta) + Y0*np.cos(theta)
    
    Xv,Yv = np.meshgrid(X0,Y0)
    Phi = np.arctan2(Yv,Xv)
    R = np.sqrt(Xv**2 + Yv**2)
    R = R*np.sqrt(q * np.cos(Phi+theta)**2 + np.sin(Phi+theta)**2 / q) 

    #Xv = Xv0*np.cos(theta) + Yv0*np.sin(theta)
    #Yv = Yv0*np.cos(theta) - Xv0*np.sin(theta)
    
    if 'Polar' in basis:
        fig,ax = plt.subplots(N,N, figsize=(20,20))
        ## Get the proper visualization
        ## in the matrix
        indices = []
        h=N;w=N;
        for p in xrange(N+w-1):
            for q in xrange(min(p,h-1),max(0,p-w+1)-1,-1):
                indices.append((h-1-q,p-q)) 
        k=0;
        for n in xrange(N):
            for m in xrange(-n,n+1,2):
                arr = polar_shapelets_refregier(n,m,sigma,theta=theta)(R,Phi)
                ax[indices[k]].imshow(arr,cmap=cm.bwr,vmax=1.,vmin=-0.5)
                ax[indices[k]].set_title(str(m)+','+str(n), fontsize=20)
                k+=1;
        ## Only below the main diagonal is going to be
        ## populated
        while k<N*N:
            fig.delaxes(ax[indices[k]]);k+=1; 
    elif 'XY' in basis:
        fig,ax = plt.subplots(M,N, figsize=(20,20))
        for n in xrange(N):
            for m in xrange(M):
                arr = elliptical_shapelet(m,n,sx=1.,sy=1.,theta=theta)(Xv,Yv)
                ax[m,n].imshow(arr,cmap=cm.bwr,vmax=1.,vmin=-0.5)
                ax[m,n].set_title(str(m)+','+str(n), fontsize=20)

    plt.savefig(basis + '_shapelets.png')
    plt.close()

def shapelet_decomposition(image_data,\
        f_path = '/home/',\
        N1=20,N2=20, basis = 'XY', solver = 'omp', image = None, \
        coeff_0 = None, noise_scale = None, \
        alpha_ = None, Num_of_shapelets = 28, \
        make_labels = False,  test_basis = False,\
        flag_gaussian_fit = True,\
        n_max = None,\
        column_number = 1.01, plot_decomp= False, plot_sol=False,\
        beta_array = [1.5, 2, 2.5]):

    """ 
    Do the shapelet decomposition and return the reconstructed image and coefficint vector;
    If noise_scale is zero (initial image) then return the label_array, and beta_aray
    
    Parameters:
    ----------
    image_data              : Array / mutable object in python, just to store centroid values and 
                                sigma of the noiseless image for the future decomp. If it is all 
                                zeros then data is obtained from the given image 
    f_path                  : Path string variable for saving the image decomp.
    N1,N2                   : Integer upper bounds for n and m quantum numbers respectively
    basis                   : String indicating the basis for decomposition:
                                -- XY - Standard Descartes coordinate shapelet space
                                -- Polar - Polar coordinate shapelet space
                                -- XY_Elliptical - Elllipse in XY shapelet space
                                -- Polar_Elliptical - Ellipse in Polar shapelets space
    solver                  : String indicating the solver to be used:
                                -- SVD - Singular Value Decomposition
                                -- omp - using the Orthogonal Matching Pursuit
                                -- P_2 - Standard least squares
                                -- P_1 - Lasso regularization technique
    image                   : ndarray of shape(size_X, size_Y) representing Image to be decomposed. 
                                If None then image is selected from the galsim RealGalaxyCatalogue
    coeff_0                 : Array of coefficients of the 0 noise decomposition
    noise_scale             : A float number which multiplies the noise_matrix
                                //used for stability tests only
    alpha_                  : A float number to be passed as *alpha* parameter to lasso solver
    Num_of_shapelets        : Integer representing the number of shapelets to be selected by OMP
                                solver
    make_labels             : Boolean to control the making of label_array, to be used in the potting
                                afterwards for appropriate labelling of shapelets:
                                -- True - look at test_basis description
                                -- False - return reconstruction, coefficients
    test_basis              : Boolean controling the return of the function:
                                -- True - return the dictionary, reconstruction, coefficients, label_array
                                -- False - return reconstruction, coefficients, label_array
    n_max                   : Ingeter nubmer indicating the number of shapelets to be used in
                                the dictionary
    column_number           : Integer number used for making distinction for images noised by 
                                differend matrices
                                // used only for stability tests
    plot_decomp, plot_sol   : Boolean flags controling the execuction of plot_solution and 
                                plot_decomposition routines from plotting_routines
    beta_array              : Array of beta values to be used for compound basis
    
    """
    
    ## Obtaining galaxy image if none is provided
    if np.all(image == None):
        ## Aquireing the noiseless image
        flag_test = True
        ## Select image 91 from the cube_real_noiseless.fits file
        image = pyfits.getdata('../../data/cube_real_noiseless.fits')[91]
        ## This is the background value for the galsim array images
        background = 1.e6*0.16**2
        image -= background
        size_X = image.shape[0]; size_Y = image.shape[1] 
    else:
        ## If an image is provided then procede here 
        size_X = image.shape[0]; size_Y = image.shape[1]

    X = np.linspace(0,size_X-1,size_X)  
    Y = np.linspace(0,size_Y-1,size_Y) 
         
    ## Just for checking plot the chosen image
    if noise_scale == 0:
        if (os.path.isfile('Plots/Initial_image_stability.png')):
            pass
        else:
            from pylab import imshow
            imshow(image)
            plt.savefig('Plots/Initial_image_stability.png')
            plt.clf() 
            plt.close()

    img_galsim = galsim.Image(image,scale = 1.0, xmin=0,ymin=0)

    ## Here catch an exception and exit the function if the FindAdaptiveMom doesn't converge
    try:
        shape = img_galsim.FindAdaptiveMom() #strict = False, watch out for failure, try block
    except RuntimeError as error:
        print("RuntimError: {0}".format(error))
        if noise_scale == 0:
            return [None]*5
        else:
            return [None]*2
    
    ## Remember this from the 0 noise image
    if np.all(image_data == 0):
        x0, y0, sigma, theta, q = get_moments(shape) 
        image_data[0] = x0; image_data[1] = y0; image_data[2] = sigma
        image_data[3] = theta; image_data[4] = q
        ## In order for function calls to be consistent
        ## Make this array even if the basis is not 'Compound'
        ## To enable correct plotting
        if ('Compound' in basis):
            beta_array = [sigma/4.,sigma/2., sigma, 2*sigma, 4*sigma]
        else:
            beta_array = [sigma]
    else:
        x0 = image_data[0]
        y0 = image_data[1]
        sigma = image_data[2] 
        theta = image_data[3]
        q = image_data[4]

    ## Initialize the basis matrix size and base_coefs sizes according
    ## to the basis used
    if ('XY' in basis):
        if ('Compound' in basis):
            ## D must be this size because different betas are included
            D = np.zeros((size_X*size_Y, len(beta_array)*N1*N2))
            ## base_coefs must be this size for the representation
            base_coefs = np.zeros((N1,N2*len(beta_array)))
        else:
            D = np.zeros((size_X*size_Y,N1*N2)) # alloc for Dictionary
            base_coefs = np.zeros((N1,N2))
    elif ('Polar' in basis):
        if ('Compound' in basis):
            D = np.zeros((size_X*size_Y,N1*N2*len(beta_array)))
            base_coefs = np.zeros(N1*N2*len(beta_array))
        else:
            D = np.zeros((size_X*size_Y,N1*N2))
            base_coefs = np.zeros(N1*N2)           

    if n_max == None:
        ## So that it is a complete triangle
        ## Formula for a symmetric triangle is n * (n+2)
        n_max = Num_of_shapelets

    signal = image.flatten() 
    shapelet_reconst = np.zeros((len(beta_array), size_X*size_Y))
    residual = np.zeros((len(beta_array), size_X*size_Y))
    residual_energy_fraction = np.zeros(len(beta_array))
    recovered_energy_fraction = np.zeros(len(beta_array))
    ## -------------------------------------------------------------
    ## Labeling could also be done inside the plot_stability routine
    ## just take the index of coeff_stability:
    ## k --> n = k/N1, m = k%N1 for cartesian
    ## k --> n,m = from indices in the adapted pascal triangle for polar
    ## -------------------------------------------------------------
    label_arr = np.chararray(D.shape[1], itemsize=10); label_arr[:]='';

    ## Decompose into Polar/Polar_Elliptical / XY /XY_Elliptical / Compound / w/ inner product
    if (basis == 'Polar') or (basis == 'Polar_Elliptical'):
        shapelet_reconst[0] = decompose_polar(basis,\
                D,base_coefs,\
                shapelet_reconst[0], signal, make_labels, \
                label_arr,\
                n_max, N1,N2,\
                x0,y0,sigma,\
                X,Y,\
                q=q, theta = theta)
    elif (basis == 'XY') or (basis == 'XY_Elliptical'):
        shapelet_reconst[0] = decompose_cartesian(basis,\
                D,base_coefs,\
                shapelet_reconst[0], signal, make_labels, \
                label_arr,\
                n_max,N1,N2,\
                x0,y0,sigma,\
                X,Y,\
                q=q,theta = theta)

    elif (basis == 'Compound_XY') or (basis == 'Compound_Polar'):
        
        shapelet_reconst = decompose_compound(basis,\
                D,base_coefs,beta_array, \
                shapelet_reconst, signal, make_labels, \
                label_arr,\
                n_max, N1,N2,\
                x0,y0,\
                X,Y,\
                polar_basis = 'refregier',\
                q=q, theta = theta)
    
    for i in xrange(len(beta_array)):
        residual[i]= signal - shapelet_reconst[i]
        residual_energy_fraction[i] = np.sum(residual[i]**2)/np.sum(signal**2)
        recovered_energy_fraction[i] = np.sum(shapelet_reconst[i]**2)/np.sum(signal**2)

    if 'Polar' in basis:
        print "Comparing moments_amp to base_coefs[0]: ", \
                np.abs(base_coefs[0]), shape.moments_amp
        print "Base coefficients sum over signal", \
            (np.sum(base_coefs**2))/(np.sum(signal**2)), \
            (np.sum(residual**2)/np.sum(signal**2)) 
    else:
        print "Comparing moments_amp to base_coefs[0,0]", \
                np.abs(base_coefs[0,0]), shape.moments_amp
        print "Base coefficients sum over signal", \
            (np.sum(base_coefs**2))/(np.sum(signal**2)), \
            (np.sum(residual**2)/np.sum(signal**2)) 

    ## Make the strings for nice representation in the output
    if noise_scale != None:
        noise_scale_str = str("%.3e" % (noise_scale))
    else:
        noise_scale_str = ""

    if (alpha_ != None):
        alpha_str = str("%.3e" % (alpha_))

    mkdir_p(f_path + 'Decomp/')
    if (plot_decomp == True):
        
        file_path_check = \
                f_path + \
                'Decomp/' + basis +'_'\
                +str(n_max) + '_' + str(column_number) +'_.png'
        
        ## Check if there is already a decomp plot
        if not(os.path.isfile(file_path_check)):

            plot_decomposition(basis, image, size_X, size_Y, \
                    base_coefs, N1, N2,\
                    shapelet_reconst, signal, \
                    residual, residual_energy_fraction ,recovered_energy_fraction, \
                    f_path + 'Decomp/_' + basis +'_'\
                    +str(n_max) + '_' + str(column_number),\
                    beta_array = beta_array)

    reconst, coeffs = select_solver_do_fitting_plot(\
            f_path, basis, coeff_0, noise_scale, \
            N1,N2,n_max,column_number,\
            image,D,signal,solver, beta_array,\
            Num_of_shapelets = Num_of_shapelets, alpha_ = alpha_, \
            flag_gaussian_fit = flag_gaussian_fit, plot = plot_sol)

    ## Check the shape data for the reconstructed image
    reconst_galsim = galsim.Image(reconst, scale =1.0, xmin=0, ymin=0)
    try:
        reconst_shape = reconst_galsim.FindAdaptiveMom()
        print "Shape of reconstruction"
        reconst_x0, reconst_y0, reconst_sigma, reconst_theta, reconst_q = get_moments(reconst_shape)
        print "x0\ty0\n"
        print reconst_x0,'\t',reconst_y0,'\n'
        print "sigma\ttheta\tq\n"
        print reconst_sigma,'\t', reconst_theta,'\t', reconst_q,'\n' 
    except RuntimeError as error:
        print "RuntimeError {0}".format(error)
        pass 

    if make_labels == True:
        if (test_basis):
            return D,reconst, coeffs, label_arr
        else:
            return reconst, coeffs, label_arr
    else:
        return reconst, coeffs


if __name__ == "__main__":   
    
    show_some_shapelets(theta = 0.)
    #p_shapelet.plot_shapelets(6,2,1)
    #check_orthonormality()
