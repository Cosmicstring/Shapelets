import sys,os
import numpy as np
import numpy.linalg as linalg
from scipy.special import hermitenorm
from scipy.integrate import quad
from sklearn import linear_model

# About the warning of the a lot of images
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits
import galsim
import math

import pdb; pdb.set_trace()

#------------------
# berry == Berry et al. MNRAS 2004
# refregier == Refregier MNRAS 2003 / Shapelets I and II
#------------------

from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
import trial_codes.polar_shapelet_decomp as p_shapelet

# Set up global LaTeX parsing
plt.rc('text', usetex=True)
plt.rc('font', **{'family' : "sans-serif"})
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', \
                r'\usepackage[utf8]{inputenc}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

DEBUG = 0

def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    """

    from errno import EEXIST
    from os import makedirs,path
    import os

    root_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    
    try:
        makedirs(root_path + mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(root_path + mypath):
            pass
        else: raise


##Define orthonormal basis - shapelets
def shapelet1d(n,x0=0,s=1):

    """Make a 1D shapelet template to be used in the construction of 2D shapelets

    @param n Energy quantum number
    @param x0 centroid
    @param s same as beta parameter in refregier

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
    
    @param n Energy quatum number
    @param m Magnetic quantum number
    @param x0 image centroid - X coordinate
    @param y0 image centroid - Y coordinate
    @param sx beta scale for the X shapelet space
    @param xy beta scale for the Y shapelet space 
    
    """

    u = lambda x: (x-x0)/sx
    v = lambda y: (y-y0)/sy
    fn = lambda x,y: np.outer(shapelet1d(m)(u(x)),shapelet1d(n)(v(y)))
    return fn

def elliptical_shapelet(m,n,x0=0,y0=0,sx=1,sy=1,theta=0):
    
    """ Make elliptical shapelets to be used in the decomposition of images later on

    @param n Energy quantum number
    @param m Magnetic quantum number
    @param x0 image centroid - X coordinate
    @param y0 image centroid - Y coordinate
    @param sx beta scale for X shapelet space (?)
    @param sy beta scale for Y shapelet space (?)
    @param theta true anomaly
    
    """
    u = lambda x,y: (x-x0)*np.cos(theta)/sx + (y-y0)*np.sin(theta)/sy
    v = lambda x,y: (y-y0)*np.cos(theta)/sy - (x-x0)*np.sin(theta)/sx
    fn = lambda x,y: np.outer(shapelet1d(m)(u(x,y)),shapelet1d(n)(v(x,y)))
    return fn

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
    
def coeff_plot2d(coeffs,N1,N2,ax=None,fig=None,orientation='vertical'):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
      fig, ax = plt.subplots()

    coeffs_reshaped = None

    if (coeffs.shape != (N1,N2)):
        coeffs_reshaped = coeffs.reshape(N1,N2) 
    else:
        coeffs_reshaped = coeffs

    #coeffs_reshaped /= coeffs_reshaped.max()
    im = ax.imshow(coeffs_reshaped,cmap=cm.bwr,interpolation='none')
    
    ## Force colorbars besides the axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%')    

    fig.colorbar(im,cax=cax,orientation=orientation)
    return fig,ax

def coeff_plot_polar(coeffs, N1,N2, len_beta = 1, N_range = 10,ax = None, fig = None, colormap = cm.bwr,\
        orientation = 'vertical'):
    """
    Plot the values of coeffs in the triangular grid
    """
    import matplotlib as mpl
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ## Initialize the array for the colors / intenzities of the coefs
    color_vals = []

    ## For now 15 is the selected N_range
    x = []
    y = []
    
    ## Check if order of n in basis (N1 currently) is smaller than N_range
    if N1 < N_range:
        N_range = N1
    
    k = 0
    for n in xrange(N_range):
        for m in xrange(-n,n+1,2):
            x.append(n)
            color_vals.append(coeffs[k])
            k += 1
            ## Make appropriate y coord
            ## So that the squares don't overlap
            ## and there is no white space between
        y.append(np.linspace(-n/2.,n/2.,n+1))
    
    ## Merge everything into one array of the same shape as x
    x = np.asarray(x)
    y = np.concatenate(y[:])    
    color_vals = np.asarray(color_vals)
    ## Control the size of squares
    dx = [x[1]-x[0]]*len(x) 
    
    ## Get the range for the colorbar
    norm = mpl.colors.Normalize(
        vmin=np.min(color_vals),
        vmax=np.max(color_vals))

    ## choose a colormap
    c_m = colormap

    ## create a ScalarMappable and initialize a data structure
    s_m = cm.ScalarMappable(cmap=c_m, norm=norm); s_m.set_array([])

    if fig == None:
        fig, ax = plt.subplots()
    
    ax.set_ylabel('m')
    ax.set_xlabel('n')
    ax.set_aspect('equal')
    ax.axis([min(x)-1., max(x)+1., min(y)-1., max(y)+1.])
    
    ## Normalize the values so that it is easier to plot
    color_vals = color_vals / np.max(color_vals)

    ## Add the coeffs as squares
    ## without any white spaces remaining
    for x,y,c,h in zip(x,y,color_vals,dx):
        ax.add_artist(\
                Rectangle(xy=(x-h/2., y-h/2.),\
                linewidth=3, color = colormap(c),\
                width=h, height=h))
    
    ## Locate the axes for colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%")

    fig.colorbar(s_m, cax = cax)
    
    return fig, ax

def show_some_shapelets(M=4,N=4):
    fig,ax = plt.subplots(M,N)
    X = np.linspace(-8,8,17)
    Y = np.linspace(-8,8,17)
    for m in xrange(M):
      for n in xrange(M):
        arr = shapelet2d(m,n)(X,Y)
        ax[m,n].imshow(arr,cmap=cm.bwr,vmax=1.,vmin=-0.5)
        ax[m,n].set_title(str(m)+','+str(n))
    plt.show()

def plot_decomposition(cube, img_idx, size_X, size_Y, \
        base_coefs,N1,N2,shapelet_reconst, signal, residual,\
        residual_energy_fraction ,recovered_energy_fraction, Path,\
        beta_array = []):

    """ Plot the decomposition obtained with the chosen __solver__

    @param cube array of images obtained from the .fits file
    @param img_idx image index in the array
    @param base_coefs base coefficients obtained from the decomposition
    @param N1,N2 number of coefficients used for n and m numbers respectively
    @param shapelet_reconst reconstruction of the image with the obtained base_coefs
    @param signal an image vector, obtained from flattening the original image matrix
    @param residual residual obtained with difference between signal and shapelet_reconst
    @param residual_energy_fraction energy fraction of the residual image
    @param recovered_energy_fraction energy fraction of the obtained image with shapelet_reconst
    @param basis variable which controls the selected __basis__ in which decomposition was made
    @param beta_array Array with betas used // Added for the compound basis

    """ 
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if basis =='Compound':
        base_coefs = base_coefs.reshape(N1,N2*len(beta_array))
    elif basis == 'XY' or basis == 'Elliptical':
        base_coefs = base_coefs.reshape(N1,N2)

    for i in xrange(len(beta_array)):
       
        str_beta = str("%.3f" % (beta_array[i]))

        if basis != 'Polar':
            left_N2 = 0 + i*N2
            right_N2 = N2 + i*N2
            coefs = base_coefs[:N1, left_N2:right_N2]
        else:
            coefs = base_coefs
            
        if np.count_nonzero(coefs) != 0:

            print 'beta ', beta_array[i], 'Decomp ', coefs.shape

            fig, ax = plt.subplots(2,2, figsize = (10, 10))
            if basis == 'XY' or basis == 'Elliptical' or basis == 'Compound':
                coeff_plot2d(coefs,N1,N2,ax=ax[1,1],fig=fig)
            elif basis == 'Polar':
                coeff_plot_polar(coefs,N1,N2,ax=ax[1,1], fig=fig)

            vmin, vmax = min(shapelet_reconst.min(),signal.min()), \
                    max(shapelet_reconst.max(),signal.max())

            im00 = ax[0,0].imshow(cube[img_idx],vmin=vmin,vmax=vmax)
            im01 = ax[0,1].imshow(shapelet_reconst.reshape(size_X,size_Y),vmin=vmin,vmax=vmax)
            im10 = ax[1,0].imshow(residual.reshape(size_X,size_Y))

            # Force the colorbar to be the same size as the axes
            divider00 = make_axes_locatable(ax[0,0])
            cax00 = divider00.append_axes("right", size="5%")

            divider01 = make_axes_locatable(ax[0,1])
            cax01 = divider01.append_axes("right", size="5%")

            divider10 = make_axes_locatable(ax[1,0])
            cax10 = divider10.append_axes("right", size="5%")

            fig.colorbar(im00,cax=cax00); fig.colorbar(im01,cax=cax01); fig.colorbar(im10,cax=cax10)
            ax[0,0].set_title('Original (noisy) image')
            ax[0,1].set_title('Reconstructed image - Frac. of energy = '\
                    +str(np.round(recovered_energy_fraction,4)))
            ax[1,0].set_title('Residual image - Frac. of energy = '\
                    +str(np.round(residual_energy_fraction,4)))
            ax[1,1].grid(lw = 1)
            ax[1,1].set_title('Magnitude of coefficients')
            fig.suptitle('Shapelet Basis decomposition')
            
            fig.tight_layout()
            plt.savefig(Path + '_' + str_beta + '_.png')
            plt.clf()

def plot_solution(N1,N2,cube, img_idx,size_X, size_Y,\
        reconst, residual, coefs_initial,\
        recovered_energy_fraction, residual_energy_fraction, \
        n_nonzero_coefs, noise_scale, Path,\
        beta_array = []):

    """ Plot obtained images from the coefficients obtained with the selected __solver__
    
    @param cube array of images obtained from the .fits file
    @param img_idx image index in the array
    @param base_coefs base coefficients obtained from the decomposition
    @param N1,N2 number of coefficients used for n and m numbers respectively
    @param shapelet_reconst reconstruction of the image with the obtained base_coefs
    @param signal an image vector, obtained from flattening the original image matrix
    @param residual residual obtained with difference between signal and shapelet_reconst
    @param residual_energy_fraction energy fraction of the residual image
    @param recovered_energy_fraction energy fraction of the obtained image with shapelet_reconst
    @param basis variable which controls the selected __basis__ in which decomposition was made
    @param n_nonzero_coefs nonzero coefficients in the coefs variable
    @param fig figure object forwarded from plot_decomposition
    @param Path to control the __savefig__ path

    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    ## Divide the initial matrix to 
    ## submatrices which correspond to the different basis
    
    if basis =='Compound':
        coefs_initial = coefs_initial.reshape(N1,N2*len(beta_array))
    elif basis == 'XY' or basis == 'Elliptical':
        coefs_initial = coefs_initial.reshape(N1,N2)
    
    for i in xrange(len(beta_array)):
       
        str_beta = str("%.3f" % (beta_array[i]))

        if basis != 'Polar':
            left_N2 = 0 + i*N2
            right_N2 = N2 + i*N2
            coefs = coefs_initial[:N1, left_N2:right_N2]
        else:
            coefs = coefs_initial
        
        if np.count_nonzero(coefs) != 0:
            print 'beta ', beta_array[i], 'shape: ', coefs.shape


            fig2, ax2 = plt.subplots(2,2, figsize = (10,10))
            vmin, vmax = min(reconst.min(),cube[img_idx].min()), \
                    max(reconst.max(),cube[img_idx].max())
            
            im00 = ax2[0,0].imshow(cube[img_idx], aspect = '1', vmin=vmin, vmax=vmax)
            im01 = ax2[0,1].imshow(\
                    reconst.reshape(size_X,size_Y), aspect = '1', vmin=vmin, vmax=vmax)
            im10 = ax2[1,0].imshow(residual.reshape(size_X,size_Y), aspect = '1')

            if basis == 'XY' or basis == 'Elliptical' or basis == 'Compound':
                coeff_plot2d(coefs,N1,N2,ax=ax2[1,1],fig=fig2) 
            elif basis == 'Polar':
                coeff_plot_polar(coefs,N1,N2, ax=ax2[1,1],fig=fig2)     

            ax2[1,1].grid(lw=1)
            
            # Force the colorbar to be the same size as the axes
            divider00 = make_axes_locatable(ax2[0,0])
            cax00 = divider00.append_axes("right", size="5%")

            divider01 = make_axes_locatable(ax2[0,1])
            cax01 = divider01.append_axes("right", size="5%")

            divider10 = make_axes_locatable(ax2[1,0])
            cax10 = divider10.append_axes("right", size="5%")  
            
            fig2.colorbar(im00,cax=cax00);
            fig2.colorbar(im01,cax=cax01); fig2.colorbar(im10,cax=cax10)
            ax2[0,0].set_title('Original (noisy) image'); 
            ax2[0,1].set_title('Reconstructed image - Frac. of energy = '\
                    +str(np.round(recovered_energy_fraction,4)))
            ax2[1,0].set_title('Residual image - Frac. of energy = '\
                    +str(np.round(residual_energy_fraction,4))); 
            fig2.suptitle('Sparse decomposition from an semi-intelligent Dictionary')

            if (noise_scale == 0):
                ax2[1,1].set_title('Magnitude of coefficients - '\
                        + str(n_nonzero_coefs) \
                        + '\n' \
                        + 'beta - ' + str_beta)
            else:
                ax2[1,1].set_title('Rel. diff in magnitude ' \
                        + r'$\displaystyle \left|\frac{N.C_i}{O.C_i} - 1 \right|$'\
                        + ' - ' + str(n_nonzero_coefs) \
                        + '\n' \
                        + 'beta - ' + str_beta)
            
            fig2.tight_layout()
            plt.savefig(Path + '_' + str_beta + '_.png')
            plt.clf()

def sparse_solver(D, signal, N1,N2, Num_of_shapelets = None):

    """ Find appropriate weights for the basis coefficients 
    obtained by the inner product routine in __shapelet_decomposition__
    using the Orthogonal Matching Pursuit algorithm

    @param D basis coefficient matrix; columns contain basis vectors
    @param signal original image to be decomposed into shapelet basis
    @param N1,N2 number of n and m quantum numbers respectively
    
    """
    n_nonzero_coefs = Num_of_shapelets
    if Num_of_shapelets == None:
        n_nonzero_coefs = N1*N2/4
    omp = OMP(n_nonzero_coefs)
    omp.fit(D,signal)
    sparse_coefs = omp.coef_
    sparse_idx = sparse_coefs.nonzero()
    sparse_reconst = np.dot(D,sparse_coefs)
    sparse_residual = signal - sparse_reconst

    residual_energy_fraction = np.sum(sparse_residual**2)/np.sum(signal**2)
    recovered_energy_fraction = np.sum(sparse_reconst**2)/np.sum(signal**2)

    return sparse_coefs, sparse_reconst, sparse_residual, \
            residual_energy_fraction, recovered_energy_fraction, n_nonzero_coefs

def solver_SVD(D, n_nonzero, signal):

    """ Find appropriate coefficients for basis vectors contained in D, reconstruct image,
    calculate residual and residual and energy fraction using the Singular Value Decomposition
    
    @param D basis coefficient matrix
    @param signal original image

    """

    rows_SVD, columns_SVD = np.shape(D)
    U, s, VT = linalg.svd(D, full_matrices = True, compute_uv = True)    
  
    ## Count in only n_nonzero signular values
    s = s[-n_nonzero:]
    
    ## In the docs it is said that the matrix returns V_transpose and not V 
    V = VT.transpose() 
    
    ## Make 1 / sigma_i array, where sigma_i are the singular values obtained from SVD
    dual_s = 1./s

    ## Initialize diagonal matrices for singular values
    S = np.zeros(D.shape)
    S_dual = np.zeros(D.shape)
    
    ## Put singular values on the diagonal
    for i in xrange(len(s)):
        S[i,i] = s[i]
        S_dual[i,i] = dual_s[i]
    
    coeffs_SVD = np.dot(V, np.dot(S_dual.transpose(), np.dot(U.transpose(),signal)))

    n_nonzero_coefs_SVD = np.count_nonzero(coeffs_SVD)
    reconstruction_SVD = np.dot(D,coeffs_SVD)
    residual_SVD = signal - reconstruction_SVD
    residual_energy_fraction_SVD = np.sum(residual_SVD**2)/np.sum(signal**2)
    recovered_energy_fraction_SVD = np.sum(reconstruction_SVD**2)/np.sum(signal**2)
    
    return coeffs_SVD, reconstruction_SVD, residual_SVD, \
            residual_energy_fraction_SVD, recovered_energy_fraction_SVD, n_nonzero_coefs_SVD

def solver_lstsq(D, signal):

    """Find appropriate coefficients for the basis matrix D, reconstruct the image, calculate
    residual and energy and residual fraction using the Orthogonal Matching Pursuit algorithm
    
    @param D basis coefficient matrix
    @param signal original image
    """
    
    coeffs_lstsq = linalg.lstsq(D, signal)[0]  
    n_nonzero_coefs_lstsq = np.count_nonzero(coeffs_lstsq) 
    reconstruction_lstsq = np.dot(D,coeffs_lstsq)
    residual_lstsq = signal - reconstruction_lstsq
    residual_energy_fraction_lstsq = np.sum(residual_lstsq**2)/np.sum(signal**2)
    recovered_energy_fraction_lstsq = np.sum(reconstruction_lstsq**2)/np.sum(signal**2)
    
    return coeffs_lstsq, reconstruction_lstsq, residual_lstsq, \
            residual_energy_fraction_lstsq, recovered_energy_fraction_lstsq, n_nonzero_coefs_lstsq

def solver_lasso_reg(D, signal, alpha = None):
    
    """Find appropriate basis coefficients for the vectors inside basis matrix D
    reconstruct the image, calculate residual and energy and residual fraction with 
    the Lasso regularization technique minimizing the L_1 norm
    
    @param D basis coefficient matrix
    @param signal original image

    """
    if (alpha == None):
        alpha = 0.001
    
    lasso_fit = linear_model.Lasso(alpha = alpha, max_iter=10000, fit_intercept = False).fit(D, signal)
    coeffs_lasso = lasso_fit.coef_
    reconstruction_lasso = np.dot(D, coeffs_lasso)
    residual_lasso = signal - reconstruction_lasso
    residual_energy_fraction_lasso = np.sum(residual_lasso**2)/np.sum(signal**2)
    recovered_energy_fraction_lasso = np.sum(reconstruction_lasso**2)/np.sum(signal**2)
    n_nonzero_coefs_lasso = np.count_nonzero(coeffs_lasso)

    return coeffs_lasso, reconstruction_lasso, residual_lasso, \
            residual_energy_fraction_lasso, recovered_energy_fraction_lasso, n_nonzero_coefs_lasso

def asses_diff(new_coefs, old_coefs):
    """
    Calculate the relative difference in coefficients
    """
    diff_arr = np.zeros_like(old_coefs)
    for i in xrange(len(old_coefs)):
        if old_coefs[i] != 0:
            diff_arr[i] = np.abs(new_coefs[i]/old_coefs[i] - 1.)
        else:
            diff_arr[i] = 0
    return diff_arr

def get_max_order(basis, n_max):
    """
    Calculating max order of shapelets included in order to get
    a symmetric triangle
    
    @param basis Basis of the decomp.
    @param n_max Chosen max number by user

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


def decompose_cartesian(x0,y0,sigma,X,Y,basis, a=1., b=1.):
    
    for k in xrange(N1*N2):
        m,n = k/N1, k%N1
        if (m+n <= get_max_order(basis,n_max)): 
            if noise_scale == 0:
                label_arr.append(str("(%d, %d)" % (n,m)))
            if basis == 'XY':
                arr = shapelet2d(m,n,x0=x0,y0=y0,sx=sigma,sy=sigma)(X,Y).flatten() 
            elif basis == 'Elliptical':
                arr = elliptical_shapelet(m,n,x0,y0, sx=a, sy=b)(X,Y).flatten()
            D[:, k] = arr
            arr_norm2 = np.dot(arr,arr)
            coef = np.dot(arr,signal)
            if(coef==0):
                base_coefs[n,m] =0
            else:
                base_coefs[n,m] = coef/np.sqrt(arr_norm2)
                shapelet_reconst = shapelet_reconst + (coef*arr)/arr_norm2

    return D,base_coefs, shapelet_reconst, label_arr

def select_solver_do_fitting_plot(f_path, coeff_0, noise_scale, solver = 'sparse',\
        Num_of_shapelets = 10, alpha_ = 0.0001):
    
    ## Default for foler path
    folder_path_word = ""
    mid_path_word = ""
    end_word = ""
    mid_name_word = '_solution_'+noise_scale_str+'_'\
            +str(N1)+'_'+str(N2)\
            +'_'+str(n_max)+'_'+basis+'_'+ str(column_number) + '_'

    ## Sparse solver
    if (solver == 'sparse'):
        
        ## Include symmetric number of shapelets
        ## symmetrize the number chosen
        Num_of_shapelets = sum_max_order(basis,(get_max_order(basis,Num_of_shapelets))) 

        coeffs, reconst, residual, \
            residual_energy_fraction, recovered_energy_fraction, \
            n_nonzero_coefs = sparse_solver(D, signal, N1,N2,Num_of_shapelets)
         
        mid_path_word = str(Num_of_shapelets) + '_' + basis
        end_word = str(Num_of_shapelets)
        folder_path_word = f_path + solver + '/' + mid_path_word +'/'

    ## SVD solver // following berry approach
    elif (solver == 'svd'):
        
        coeffs, reconst, residual, \
        residual_energy_fraction, recovered_energy_fraction, \
        n_nonzero_coefs = solver_SVD(D, n_max, signal) 

        folder_path_word = f_path + solver + '/'

    ## Ordinary least squares solver
    elif (solver == 'lstsq'):  

        coeffs, reconst, residual, \
        residual_energy_fraction, recovered_energy_fraction, \
        n_nonzero_coefs = solver_lstsq(D, signal) 

        folder_path_word = f_path + solver + '/' 
    
    elif (solver == 'lasso'): #This is with the Lasso regularization
           
        coeffs, reconst, residual, \
            residual_energy_fraction, recovered_energy_fraction, \
            n_nonzero_coefs = solver_lasso_reg(D, signal, alpha_)
    
        mid_path_word = str("%.3e" % (alpha_))
        folder_path_word = f_path + solver + '/' + mid_path_word + '/'           

        
    if (noise_scale == 0):
        coefs_plot = coeffs
    else:
        coefs_plot = asses_diff(coeffs,coeff_0)

    ## Make a dir for storage of decompositions 
    mkdir_p(folder_path_word)

    if plot_decomp == True:
        plot_solution(N1,N2,cube,img_idx,size_X,size_Y, \
            reconst, sparse_residual, coefs_plot,\
            recovered_energy_fraction, residual_energy_fraction, n_nonzero_coefs, \
            noise_scale, \
            f_path + solver + '/'+ mid_path_word +'/' \
            + solver + mid_name_word + end_word,\
            beta_array = beta_array)
        
    return reconst.reshape(size_X,size_Y),coefs


def shapelet_decomposition(image_data,\
        f_path = '/home/',\
        N1=20,N2=20, basis = 'XY', solver = 'sparse', image = None, \
        coeff_0 = None, noise_scale = None, alpha_ = None, Num_of_shapelets = None, n_max = None,\
        column_number = 1.01, plot_decomp= False, \
        q = 2., beta_array = [1.5, 2, 2.5]):

    """ 
    Do the shapelet decomposition
    
    @param image_data Array / mutable object in python, just to store centroid values and sigma
                      of the noiseless image for the future decomp.
    @param f_path Path variable for saving the image decomp.
    @param N1,N2 n and m quantum numbers respectively
    @param basis do decomposition in this basis
        -- XY - Standard Descartes coordinate shapelet space
        -- Polar - Polar coordinate shapelet space
        -- Elliptical - Elliptical coordinate shapelet space
    @param solver choose an algorithm for fitting the coefficients
        -- SVD - Singular Value Decomposition
        -- sparse - using the Orthogonal Matching Pursuit
        -- P_2 - Standard least squares
        -- P_1 - Lasso regularization technique
    @param image Image to be decomposed
    @param coeff_0 Coefficients of the 0 noise decomposition
    @param noise_scale A number which multiplies the noise_matrix
    @param alpha_ Scalar factor in fron of the l_1 norm in the lasso_regularization method
    @param Num_of_shapelets Number which refers to maximum allowed number for OMP method to use
    @param n_max This nubmer refers to the maximum order of shapelets that could be used in decomp.
    @param column_number Just used for making distinction for images noised by differend matrices
    @param plot_decomp Should the plot_solution and plot_decomposition be used or not
    @param q Axis ratio for elliptical coordinates, defined as q = b / a
    @param theta Direction angle of the ellipse

    @returns reconstructed image and coefficients of the selected method along with labels of shapelets
    """

    ## Obtaining galaxy images
    if (image == None) or (noise_scale == 0):
        #cube = pyfits.getdata('../../data/cube_real.fits')
        ## Aquireing the noiseless image
        cube = pyfits.getdata('../../data/cube_real_noiseless.fits')
        background = 1.e6*0.16**2
        img = galsim.Image(78,78) # cube has 100, 78x78 images
        size_X = 78; size_Y = 78
        pick_an_img = [91]
    else:
        ## Added this part for stability test with noise
        ## Background is zero because I already substracted background in the first decomp
        size_X = np.shape(image)[0]; size_Y = np.shape(image)[1]
        cube = [image]
        background = 0
        pick_an_img = [0] 

    X = np.linspace(0,size_X-1,size_X)  
    Y = np.linspace(0,size_Y-1,size_Y)

    for img_idx in pick_an_img:

        ## Take the image, reduce for background, find moments set n_max

        cube[img_idx] -= background
        
        ## Just for checking plot the chosen image
        if noise_scale == 0:
            if os.path.isfile('Plots/Initial_image.png'):
                pass
            else:
                from pylab import imshow
                imshow(cube[img_idx])
                plt.savefig('Plots/Initial_image.png')
                plt.clf()
        
        img = galsim.Image(cube[img_idx],xmin=0,ymin=0)
        
        ## Here catch an exception and exit the function if the FindAdaptiveMom doesn't converge
        try:
            shape = img.FindAdaptiveMom() #strict = False, watch out for failure, try block
        except RuntimeError as error:
            print("RuntimError: {0}".format(error))
            return None, None,None
        
        ## Remember this from the 0 noise image
        if noise_scale == 0:
            x0,y0 = shape.moments_centroid.x, shape.moments_centroid.y ## possible swap b/w x,y
            sigma = shape.moments_sigma
            image_data[0] = x0; image_data[1] = y0; image_data[2] = sigma
        else:
            x0 = image_data[0]
            y0 = image_data[1]
            sigma = image_data[2]
        
        ## In order for function calls to be consistent
        ## Make this array even if the basis is not 'Compound'
        ## To enable correct plotting
        if basis == 'Compound':
            beta_array = [sigma/4.,sigma/2., sigma, 2*sigma, 4*sigma]
        else:
            beta_array = [sigma]

        if (basis == 'XY') or (basis == 'Elliptical'):
            D = np.zeros((size_X*size_Y,N1*N2)) # alloc for Dictionary
            base_coefs = np.zeros((N1,N2))
        elif (basis == 'Polar'):
            D = np.zeros((size_X*size_Y,N1*N2))#, dtype=complex)
            base_coefs = np.zeros(N1*N2)#, dtype=complex)
        elif (basis == 'Compound'):
            ## D must be this size because different betas are included
            D = np.zeros((size_X*size_Y, len(beta_array)*N1*N2))
            ## base_coefs must be this size for the representation
            base_coefs = np.zeros((N1,N2*len(beta_array)))


        if n_max == None:
            ## So that it is a complete triangle
            ## Formula for a symmetric triangle is n * (n+2)
            n_max = 21

        #Make a meshgrid for the polar shapelets

        Xv, Yv = np.meshgrid((X-x0),(Y-y0))
        R = np.sqrt(Xv**2 + Yv**2)
        
        Phi = np.zeros_like(R)
        for i in xrange(np.shape(Xv)[0]):
            for j in xrange(np.shape(Xv)[1]):
                Phi[i,j] = math.atan2(Yv[i,j], Xv[i,j])

        signal = cube[img_idx].flatten() 

        shapelet_reconst = np.zeros_like(signal)

        #Decompose into Polar or XY or Elliptical w/ inner product 
        if (basis == 'Polar'):

            #Set the counter for columns of basis matrix D

            polar_basis = 'refregier' 
            label_arr = []
            k_p = 0
            ## For symmetric coeffs triangle solve
            ## n^2 + 2n + 2 = 2n_max, positive solution, such that
            ## Sum(i+1)_{i in [0,n]} <= n_max
                        
            for n in xrange(get_max_order(basis,n_max)):
                for m in xrange(-n,n+1,2):
                    
                    ## n_max - defined as:
                    ## theta_max (galaxy size) / theta_min (smallest variation size) - 1  

                    if noise_scale==0:
                        label_arr.append(str("(%d, %d)" % (n,m)))
                    
                    if (polar_basis == 'refregier'):
                        arr_res = \
                                p_shapelet.polar_shapelets_refregier(n,m,sigma)(R,Phi).flatten() 
                        arr = arr_res
                        arr_im = arr_res.imag
                    elif (polar_basis == 'berry'):
                        arr_res = p_shapelet.polar_shapelets_berry(n,m,sigma)(R,Phi).flatten()
                        arr = arr_res 
                        arr_im = arr_res.imag
                    ## Make the basis matrix D
                    D[:,k_p] = arr_res 
                    k_p += 1
                    ## Calculate the norms of basis vectors and coefficients
                    arr_norm2_res = np.dot(arr_res,arr_res)
                    arr_norm2 = np.dot(arr, arr)
                    arr_norm_im2 = np.dot(arr_im, arr_im)
                    coef_r = np.dot(arr,signal)
                    coef_im = np.dot(arr_im, signal)
                    coef_res= np.dot(arr_res, signal)

                    ## Add coefficients to basis_coefs array for later use
                    ## Make the shapelet reconstruction
                    if (coef_res==0): 
                        base_coefs[k_p] = 0 
                    else: 
                        base_coefs[k_p] = coef_res/np.sqrt(arr_norm2_res)
                        shapelet_reconst = shapelet_reconst \
                                + coef_res*arr_res/arr_norm2_res
        elif(basis == 'XY'):
            
             label_arr = []
             for k in xrange(N1*N2):
                m,n = k/N1, k%N1 
                ## For symmetric coeffs / complete triangle
                ## such that Sum(i + 1)_{i in [0,n]} <= n_max
                ## solve n^2 + 2n + 2 = 2n_max                 
                if (m+n <= get_max_order(basis,n_max)): 
                    if noise_scale==0:
                        label_arr.append(str("(%d, %d)" % (m,n)))
                    arr = shapelet2d(m,n,x0=x0,y0=y0,sx=sigma,sy=sigma)(X,Y).flatten() 
                    D[:,k] = arr
                    arr_norm2 = np.dot(arr, arr)
                    coef = np.dot(arr,signal)
                    if(coef==0): 
                        base_coefs[n,m] = 0
                    else: 
                        base_coefs[n,m] = coef/np.sqrt(arr_norm2)                        
                        shapelet_reconst = shapelet_reconst + (coef*arr)/arr_norm2 
        elif(basis == 'Elliptical'):
            ## Params of ellipse (major and minor) defined throug:
            ## P_ellipse = P_circle, i.e. beta^2 = ab, and with provided ratio q = b/a
            a = sigma / np.sqrt(q)
            b = sigma * np.sqrt(q)
            label_arr =[]
            
            for k in xrange(N1*N2):
                m,n = k/N1, k%N1
                if (m+n <= get_max_order(basis,n_max)): 
                    if noise_scale == 0:
                        label_arr.append(str("(%d, %d)" % (n,m)))
                    arr = elliptical_shapelet(m,n,x0,y0, sx=a, sy=b)(X,Y).flatten()
                    D[:, k] = arr
                    arr_norm2 = np.dot(arr,arr)
                    coef = np.dot(arr,signal)
                    if(coef==0):
                        base_coefs[n,m] =0
                    else:
                        base_coefs[n,m] = coef/np.sqrt(arr_norm2)
                        shapelet_reconst = shapelet_reconst + (coef*arr)/arr_norm2
        elif(basis == 'Compound'):
            ## Just put different betas (range TBD) basis into a big basis matrix 
            ## let the algorithms pick the weights. Watch out that the order stays symmetric
            
            label_arr = []
            idx = 0
            step = 0
            for sigma in beta_array:
                a = sigma / np.sqrt(q)
                b = sigma * np.sqrt(q)
                for k in xrange(N1*N2):
                    ## Number of cols of D is N1*N2*len(beta_array)
                    ## beta should change when whole basis is sweeped by k
                    m,n = k/N1, k%N1
                    if (m+n <= get_max_order(basis, n_max)):
                        
                        if noise_scale == 0:
                            label_arr.append(str("(%d, %d)" % (n,m)))
                        arr = elliptical_shapelet(m,n,x0,y0,sx=a,sy=b)(X,Y).flatten()
                        D[:,idx] = arr
                        idx += 1
                        arr_norm2 = np.dot(arr,arr)
                        coef = np.dot(arr, signal)
                        if coef == 0:
                            ## Watch not to overwrite the existing 
                            base_coefs[n, m + N2*step] = 0
                        else:
                            base_coefs[n, m + N2*step] = coef / np.sqrt(arr_norm2)
                            shapelet_reconst += coef*arr/arr_norm2
                ## Basis finished increase the step
                step += 1
            
            pass

        residual= signal - shapelet_reconst
        residual_energy_fraction = np.sum(residual**2)/np.sum(signal**2)
        recovered_energy_fraction = np.sum(shapelet_reconst**2)/np.sum(signal**2)

        if basis =='Polar':
            print "Comparing moments_amp to lowest order base_coefs", \
                    np.abs(base_coefs[np.where(base_coefs!=0)[0][0]] ), shape.moments_amp
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
        noise_scale_str = str("%.3e" % (noise_scale))
        if (alpha_ != None):
            alpha_str = str("%.3e" % (alpha_))

        mkdir_p(f_path + 'Decomp/')
        if (plot_decomp == True):
            
            ## Check if there is already the XY decomp
            if os.path.isfile(f_path + \
                    'Decomp/'+ solver+ '_' + basis +'_'\
                    +str(n_max) + '_' + str(column_number) +'_.png') == False:
                if basis == 'Compound':
                    if noise_scale == 0:
                        plot_decomposition(cube, img_idx, size_X, size_Y, \
                            base_coefs, N1, N2,\
                            shapelet_reconst, signal, \
                            residual, residual_energy_fraction ,recovered_energy_fraction, \
                            f_path + 'Decomp/'+ solver+ '_' + basis +'_'\
                            +str(n_max) + '_' + str(column_number) +'_.png',\
                            beta_array = beta_array)
                else:
                    plot_decomposition(cube, img_idx, size_X, size_Y, \
                        base_coefs, N1, N2,\
                        shapelet_reconst, signal, \
                        residual, residual_energy_fraction ,recovered_energy_fraction, \
                        f_path + 'Decomp/'+ solver+ '_' + basis +'_'\
                        +str(n_max) + '_' + str(column_number) +'_.png',\
                        beta_array = beta_array)

        reconst, coeffs = select_solver_do_fitting_plot(\
                f_path, coeff_0, noise_scale, \
                Num_of_shapelets = Num_of_shapelets, alpha_ = alpha_)
        ## Sparse solver
        if (solver == 'sparse'):
            
            if Num_of_shapelets == None:
                Num_of_shapelets = 10

            ## Include symmetric number of shapelets
            ## symmetrize the number chosen
            Num_of_shapelets = sum_max_order(basis,(get_max_order(basis,Num_of_shapelets))) 

            sparse_coefs, sparse_reconst, sparse_residual, \
                residual_energy_fraction, recovered_energy_fraction, \
                n_nonzero_coefs = sparse_solver(D, signal, N1,N2,Num_of_shapelets)

            if (noise_scale == 0):
                sparse_coefs_plot = sparse_coefs
            else:
                sparse_coefs_plot = asses_diff(sparse_coefs,coeff_0)

            # Make a dir for storage of decompositions 
            mkdir_p(f_path + 'Sparse/'+str(Num_of_shapelets) +'_'+basis + '/')

            if plot_decomp == True:
                plot_solution(N1,N2,cube,img_idx,size_X,size_Y, \
                    sparse_reconst, sparse_residual, sparse_coefs_plot,\
                    recovered_energy_fraction, residual_energy_fraction, n_nonzero_coefs, \
                    noise_scale, \
                    f_path + 'Sparse/'+ str(Num_of_shapelets) + '_'+ basis+'/Sparse_solution_'\
                    +noise_scale_str+'_'+str(N1)+'_'+str(N2)+'_'+str(n_max)+'_'+basis+'_'\
                    + str(column_number)+'_'+str(Num_of_shapelets),\
                    beta_array = beta_array)

            if noise_scale == 0:
                return cube[img_idx],sparse_reconst.reshape(size_X,size_Y), sparse_coefs, \
                        label_arr, beta_array
            else:
                return sparse_reconst.reshape(size_X,size_Y), sparse_coefs   

        # SVD solver // following berry approach
        if (solver == 'SVD'):
            
            coeffs_SVD, reconstruction_SVD, residual_SVD, \
            residual_energy_fraction_SVD, recovered_energy_fraction_SVD, \
            n_nonzero_coefs_SVD = solver_SVD(D, n_max, signal)
            
            if (noise_scale == 0):
                SVD_coefs_plot = coefs_SVD
            else:
                SVD_coefs_plot = asses_diff(coefs_SVD,coeff_0)

            # Make a dir for storage of decomp.
            mkdir_p(f_path + 'SVD/')

            if plot_decomp == True:
                plot_solution(N1,N2,cube, img_idx,size_X,size_Y, \
                    reconstruction_SVD, residual_SVD, SVD_coefs_plot, \
                    recovered_energy_fraction_SVD, residual_energy_fraction_SVD, \
                    n_nonzero_coefs_SVD, noise_scale, \
                    f_path + 'SVD/SVD_solution_' \
                     +noise_scale_str+'_'+str(N1)+'_'+str(N2)+'_'+str(n_max)+'_'+basis+'_'\
                     + str(column_number),\
                     beta_array = beta_array)
            
            if noise_scale == 0:
                return cube[img_idx],sparse_reconst.reshape(size_X,size_Y), sparse_coefs, \
                        label_arr, beta_array
            else:
                return sparse_reconst.reshape(size_X,size_Y), sparse_coefs 
            

        #Ordinary least squares solver
        if (solver == 'lstsq'):  

            coeffs_lstsq, reconstruction_lstsq, residual_lstsq, \
            residual_energy_fraction_lstsq, recovered_energy_fraction_lstsq, \
            n_nonzero_coefs_lstsq = solver_lstsq(D, signal)
            
            if (noise_scale == 0):
                lstsq_coefs_plot = coeffs_lstsq 
            else:
                lstsq_coefs_plot = asses_diff(coeffs_lstsq, coeff_0)

            # Make a dir for storage of decomp.
            mkdir_p(f_path + 'Lstsq/')

            if plot_decomp == True:
                plot_solution(N1,N2,cube, img_idx,size_X,size_Y,  \
                    reconstruction_lstsq, residual_lstsq, \
                    lstsq_coefs_plot, \
                    recovered_energy_fraction_lstsq, residual_energy_fraction_lstsq, \
                    n_nonzero_coefs_lstsq, noise_scale, \
                    f_path + 'Lstsq/lstsq_solution_'\
                    +noise_scale_str+'_'+str(N1)+'_'+str(N2)+'_'+str(n_max)+'_'+basis+'_'\
                    +str(column_number),\
                    beta_array = beta_array)
            
            
            if noise_scale == 0:
                return cube[img_idx],sparse_reconst.reshape(size_X,size_Y), sparse_coefs, \
                        label_arr, beta_array
            else:
                return sparse_reconst.reshape(size_X,size_Y), sparse_coefs
        
        if (solver == 'lasso'): #This is with the Lasso regularization
            
            if alpha_ == None:
                alpha_ = 0.0001
                
            coeffs_lasso, reconstruction_lasso, residual_lasso, \
                residual_energy_fraction_lasso, recovered_energy_fraction_lasso, \
                n_nonzero_coefs_lasso = solver_lasso_reg(D, signal, alpha_)
        
            if (noise_scale == 0):
                lasso_coefs_plot = coeffs_lasso
            else:
                lasso_coefs_plot = asses_diff(coeffs_lasso, coeff_0)
                
            # Make a dir for storage of decomp.
            mkdir_p(f_path + 'Lasso/'+alpha_str+'/')
            
            if plot_decomp == True:
                plot_solution(N1,N2,cube, img_idx,size_X,size_Y,  \
                        reconstruction_lasso, residual_lasso, \
                        lasso_coefs_plot, \
                        recovered_energy_fraction_lasso, residual_energy_fraction_lasso, \
                        n_nonzero_coefs_lasso, noise_scale, \
                        f_path + 'Lasso/' + alpha_str + '/lasso_solution_'\
                        +noise_scale_str+ '_' + str(N1)+'_'+str(N2)+'_'+str(n_max)+'_'+basis+'_'\
                        + str(column_number)+alpha_str+'_',\
                        beta_array = beta_array)

            if noise_scale == 0:
                return cube[img_idx],sparse_reconst.reshape(size_X,size_Y),sparse_coefs,\
                            label_arr, beta_array
            else:
                return sparse_reconst.reshape(size_X,size_Y), sparse_coefs

def plot_stability(coeff_stability, coeff_0, N1, N2, noise_img_num, \
        n_max = 20, label_arr = None,\
        beta_array = [], signal_to_noise = None, \
        basis = 'Polar', solver = 'lasso', \
        fname_imag = 'Plots/Lasso/Stability/',\
        variance_folder = 'Plots/Lasso/Stability/',\
        mid_word = None):
    
    """
    Plot the stability matrices and variance matrices for selected solver method
    """
    ## Initialize the stability and variance arrays
    coeff_stability_res = np.zeros(coeff_stability.shape[0])
    variance = np.zeros(coeff_stability.shape[0])
    variance_sqrt = np.zeros(coeff_stability.shape[0])
    tmp = np.zeros(noise_img_num)

    ## Find the mean coefficients and the variance
    ## <N.C>_i and Var(N.C_i)
    for i in xrange(coeff_stability.shape[0]):
        for j in xrange(noise_img_num):
            coeff_stability_res[i] += coeff_stability[i,j]
            tmp[j] = coeff_stability[i,j]
        coeff_stability_res[i] = coeff_stability_res[i]/noise_img_num
        variance_sqrt[i] = np.sqrt(np.var(tmp))
        if coeff_stability_res[i] != 0:
            variance[i] = np.sqrt(np.var(tmp))/np.abs(coeff_stability_res[i])
        else:
            variance[i] = 0
    ## Asses the difference
    ## |<N.C>_i / O.C_i - 1|
    for i in xrange(len(coeff_stability_res)):
        if coeff_0[i] != 0:
            coeff_stability_res[i] = np.abs(coeff_stability_res[i]/coeff_0[i] - 1)
        else:
            coeff_stability_res[i] = 0

    ## Add the scatter plot
    ## How many points to plot  
    n_max = sum_max_order(basis, get_max_order(basis,n_max))    
    n_nonzero = np.count_nonzero(coeff_stability_res)
    n_nonzero = sum_max_order(basis,get_max_order(basis, n_nonzero))

    ## Take the biggest possible number of 
    ## coefficients
    if n_max <= n_nonzero:
        N_plot = n_max
    else:
        N_plot = n_nonzero

    ## Indices measured according to the smallest number of
    ## nonzero values in these arrays
    nonzero_var_sqrt = len(np.where(variance_sqrt != 0)[0])
    nonzero_coeff_res = len(np.where(coeff_stability_res != 0)[0])
    
    ## If you can get all of the nonzero if not
    ## then just N_plot values
    ## If N_plot > nonzero * it returns the whole set of indices
    if nonzero_var_sqrt <= nonzero_coeff_res:
        get_idx = np.where(variance_sqrt != 0)[0][:N_plot]
    else:
        get_idx = np.where(coeff_stability_res != 0)[0][:N_plot]

    coeff_res_r = coeff_stability_res[get_idx]
    variance_sqrt_r = variance_sqrt[get_idx]
    label_arr_r = np.asarray(label_arr)
    label_arr_r = label_arr_r[get_idx]

    arange_x = np.arange(len(get_idx))
    
    fig_scat, ax_scat = plt.subplots()

    plt.errorbar(arange_x, coeff_res_r, yerr=variance_sqrt_r, fmt='bo', \
            label='Coeff. value')
    plt.xticks(arange_x, label_arr_r)
    plt.title('Scatter plot of first %d coefs' % (N_plot))
    plt.legend(loc = 'best')
    plt.setp(ax_scat.get_xticklabels(),rotation=90, horizontalalignment = 'right')
    
    ax_scat.tick_params(axis='x', which ='both', pad = 10)
    
    ## Set the font of the ticks
    for tick in ax_scat.xaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    for tick in ax_scat.xaxis.get_minor_ticks():
        tick.label.set_fontsize(7)

    mkdir_p(fname_imag)
    plt.savefig(fname_imag + solver + '_' + mid_word + "_scatter_coefs.png")
    plt.clf()

    ## Calculate the relative variance of decompositions
    ## Coeff_stability is already in the needed form
    ## variance_i = Var(N.C_i)
    
    if basis =='Compound':
        variance_initial = variance.reshape(N1,N2*len(beta_array))
        variance_sqrt_initial = variance_sqrt.reshape(N1,N2*len(beta_array))
        coefs_initial = coeff_stability_res.reshape(N1,N2*len(beta_array))
    elif basis == 'XY' or basis == 'Elliptical':
        variance_sqrt_initial = variance_sqrt.reshape(N1,N2)
        variance_initial = variance.reshape(N1,N2)
        coefs_initial = coef_stability_res.reshape(N1,N2)
    
    for i in xrange(len(beta_array)):
        
        str_beta = str("%.3f" % (beta_array[i]))

        if basis != 'Polar':
            left_N2 = 0 + i*N2
            right_N2 = N2 + i*N2
            variance = variance_initial[:N1, left_N2:right_N2]
            variance_sqrt = variance_sqrt_initial[:N1, left_N2:right_N2]
            coefs = coefs_initial.reshape[:N1,left_N2:right_N2]
        else:
            variance = variance_initial
            variance_sqrt = variance_sqrt_initial
            coefs = coefs_initial

        ## Add the stability plot 
        fig, ax = plt.subplots()
        
        if basis == 'XY' or basis == 'Elliptical' or basis == 'Compound':
            coeff_plot2d(coefs,N1,N2,ax=ax,fig=fig) 
        elif basis == 'Polar':
            coeff_plot_polar(coefs,N1,N2,ax=ax,fig=fig)

        str_s_to_n = str("%.3e" % (signal_to_noise)) 

        ax.grid(lw=1)
        ax.set_aspect('equal')
        ax.set_title('Stability of coefs '\
                + r'$\displaystyle \left|\frac{<N.C.>_i}{O.C._i} - 1\right|$' \
                + '\n' \
                + 'N.C is averaged over the number of noise realizations' \
                + '\n' \
                + 'S/N = ' + str_s_to_n\
                + 'beta - ' + str_beta)

        mkdir_p(fname_imag)
        
        fig.tight_layout()
        plt.savefig(fname_imag + solver + '_stability_'+mid_word+'_'+str_s_to_n+'_'\
                +str_beta+'_.png')
        plt.clf()

        fig_var, ax_var = plt.subplots()
        
        if basis=='XY' or basis == 'Elliptical' or basis == 'Compound':
            coeff_plot2d(variance,N1,N2,ax=ax_var,fig=fig_var) 
        elif basis == 'Polar':
            coeff_plot_polar(variance,N1,N2,ax=ax_var,fig=fig_var)

        ax_var.grid(lw=1)
        ax_var.set_aspect('equal')
        ax_var.set_title('Variance matrix '\
                + r'$\displaystyle \sqrt{Var\left(N.C._i\right)} / |\left< N.C._i\right>|$' \
                + '\n' \
                + 'S/N = ' + str_s_to_n\
                + 'beta - ' + str_beta)
        
        fig_var.tight_layout()
        
        mkdir_p(variance_folder)
        plt.savefig(variance_folder + solver + '_variance_rel_'+mid_word+'_'+str_s_to_n+'_'\
                +str_beta+'_.png')
        plt.clf()
        
        ## Just the variance image
        fig_var, ax_var = plt.subplots()
        
        if basis == 'XY' or basis == 'Elliptical' or basis == 'Compound':
            coeff_plot2d(variance_sqrt,N1,N2,ax=ax_var,fig=fig_var) 
        elif basis == 'Polar':
            coeff_plot_polar(variance_sqrt,N1,N2,ax=ax_var,fig=fig_var)

        ax_var.grid(lw=1)
        ax_var.set_aspect('equal')
        ax_var.set_title('Variance matrix '\
                + r'$\displaystyle \sqrt{Var\left(N.C._i\right)}$' \
                + '\n' \
                + 'S/N = ' + str_s_to_n\
                + 'beta - ' + str_beta)
        
        fig_var.tight_layout()
        
        mkdir_p(variance_folder)
        plt.savefig(variance_folder + solver + '_variance_sqrt_'+mid_word+'_'+str_s_to_n+'_'\
                +str_beta+'_.png')
        plt.clf()

def test_stability(solver, basis, \
        noise_scale, noise_img = None, noise_img_num = 10,\
        size_X = 78, size_Y = 78, \
        alpha_ = None, Num_of_shapelets_array = None):

    ## Import the util package for the weights
    import galsim
    from utils.get_gaussian_weight_image import get_gaussian_weight_image as gen_weight_image

    ## Initialize values
    N1 = 20; N2=20; n_max = 21; Num_of_shapelets = None; alpha = None
    
    image = None; image_curr = None; coeffs_0 = None; coeffs_curr = None; coeff_stability = None
    
    image_data = np.zeros(3)

    ## Now select alphas if the method is lasso
    if solver == 'lasso':
        for l in xrange(len(alpha_)):
            
            alpha = alpha_[l]
            ## Iterate through different noise scales
            ## image - image to be decomposed
            ## image_curr - temporary storage for decomposed image
            
            # Make the no noise image
            f_path = 'Plots/' + str("%.3e" % (0.0)) + '/'
            mkdir_p(f_path)
            
            image_0, image_reconst, coeff_0, label_arr, beta_array  =\
                        shapelet_decomposition(image_data,\
                        f_path = f_path,\
                        N1=N1,N2=N2,basis=basis,solver=solver,\
                        image=None, coeff_0 =None, noise_scale=0,\
                        alpha_=alpha,\
                        n_max = n_max,\
                        plot_decomp = True)

            ## If the initial decomposition doesn't fail continue
            if (image_reconst != None):
                ## The matrix to be used for stability calculations
                ## Variance and mean difference
                coeff_stability = np.zeros((len(coeff_0),noise_img_num))
                
                ## A sample image used for the S/N calculation
                image_sample = image_0 + noise_scale*noise_img[:,0].reshape(size_X,size_Y)
                
                ## Asses the weights and calculate S/N
                weight_image,flag = gen_weight_image(image_sample)
                
                if flag == 1:
                    weight_image = weight_image.flatten()
                    signal_image = image_sample.flatten()
                    
                    signal_to_noise = (1./noise_scale) * np.dot(weight_image,signal_image) \
                            / np.sum(weight_image)
                    
                    ## Make folders for storage
                    f_path = 'Plots/' + str("%.3e" % (noise_scale)) + '/'
                    mkdir_p(f_path)

                    k = 0
                    for i in xrange(noise_img_num): 
                        
                        ## Add the noise_matrix to the 0 noise image
                        image = image_0 + noise_scale*noise_img[:,i].reshape(size_X,size_Y)
                        
                        image_reconst_curr, coeffs_curr, foo =\
                                shapelet_decomposition(image_data,\
                                f_path = f_path,\
                                N1=N1,N2=N2,basis=basis,solver=solver,\
                                image=image, coeff_0=coeff_0, noise_scale=signal_to_noise,\
                                alpha_=alpha,\
                                n_max = n_max,\
                                plot_decomp=True,\
                                column_number = i)
                        
                        ## Because of lack of convergence of AdaptiveMom of galsim
                        ## Check for None returns and append only if it is != 0
                        if coeffs_curr != None:
                                coeff_stability[:,k] = coeffs_curr
                                k += 1

                    plot_stability(coeff_stability, coeff_0, N1, N2, noise_img_num, \
                            n_max = n_max, label_arr = label_arr, \
                            beta_array = beta_array, signal_to_noise = signal_to_noise, \
                            basis = basis, solver = solver, \
                            fname_imag = f_path + 'Stability/',\
                            variance_folder = f_path + 'Stability/',\
                            mid_word = str("%.3e" % (alpha))+'_'+str(basis))
             
    elif(solver == 'sparse'):
        
        # Select number of shapelets that would be selected by OMP
        for c in xrange(len(Num_of_shapelets_array)):
            Num_of_shapelets = Num_of_shapelets_array[c]
            
            # Make the no noise image
            f_path = 'Plots/' + str("%.3e" % (0.0)) + '/'
            mkdir_p(f_path)

            image_0, image_reconst, coeff_0, label_arr, beta_array =\
                        shapelet_decomposition(image_data,\
                        f_path = f_path,\
                        N1=N1,N2=N2,basis=basis,solver=solver,\
                        image=None, coeff_0=None, noise_scale=0,\
                        Num_of_shapelets=Num_of_shapelets, \
                        plot_decomp = True,\
                        n_max = n_max)
                        
            ## If 0 noise decomp. fails don't do anything
            if (image_reconst != None):
                ## The matrix to be used for stability calculations
                ## Variance and mean difference
                coeff_stability = np.zeros((len(coeff_0),noise_img_num))
                
                ## A sample image used for the S/N calculation
                image_sample = image_0 + noise_scale*noise_img[:,0].reshape(size_X,size_Y)
                
                ## Asses the weights and calculate S/N
                weight_image,flag = gen_weight_image(image_sample)
                
                if flag == 1:
                    weight_image = weight_image.flatten()
                    signal_image = image_sample.flatten()

                    signal_to_noise = (1./noise_scale) * np.dot(weight_image,signal_image) \
                            / np.sum(weight_image)
                    
                    ## Make folders for storage
                    f_path = 'Plots/' + str("%.3e" % (noise_scale)) + '/'
                    mkdir_p(f_path)
                    
                    k=0
                    for i in xrange(noise_img_num):
                                        
                        ## Add the noise_matrix to the 0 noise image
                        image = image_0 + noise_scale*noise_img[:,i].reshape(size_X,size_Y)
                        
                        image_reconst_curr, coeffs_curr =\
                                shapelet_decomposition(image_data,\
                                f_path = f_path,\
                                N1=N1,N2=N2,basis=basis,solver=solver,\
                                image=image, coeff_0=coeff_0, noise_scale=signal_to_noise,\
                                Num_of_shapelets = Num_of_shapelets,\
                                plot_decomp = True,\
                                n_max = n_max,\
                                column_number = i)
                        
                        if coeffs_curr != None:
                            coeff_stability[:,k] = coeffs_curr
                            k+=1

                    plot_stability(coeff_stability, coeff_0, N1, N2, noise_img_num, \
                            n_max = n_max, label_arr = label_arr, \
                            beta_array = beta_array, signal_to_noise = signal_to_noise, \
                            basis = basis, solver = solver, \
                            fname_imag = f_path + 'Stability/',\
                            variance_folder = f_path + 'Stability/',\
                            mid_word = str(\
                            sum_max_order(basis,get_max_order(basis,Num_of_shapelets))) \
                            + '_' + str(basis))           
            
    elif(solver == 'lstsq'):
        
        # Make the no noise image
        f_path = 'Plots/' + str("%.3e" % (0.0)) + '/'
        mkdir_p(f_path)

        image_0, image_reconst, coeff_0, label_arr, beta_array =\
                    shapelet_decomposition(image_data,\
                    f_path = f_path,\
                    N1=N1,N2=N2,basis=basis,solver=solver,\
                    image = None,coeff_0= None, noise_scale=0,\
                    n_max = n_max,\
                    plot_decomp = True)

        if image_reconst!=None:
            # The matrix to be used for stability calculations
            # Variance and mean difference
            coeff_stability = np.zeros((len(coeff_0),noise_img_num))
            
            ## A sample image used for the S/N calculation
            image_sample = image_0 + noise_scale*noise_img[:,0].reshape(size_X,size_Y)
            ## Asses the weights and calculate S/N
            weight_image,flag = gen_weight_image(image_sample)
            
            if flag == 1:    
                weight_image = weight_image.flatten()
                signal_image = image_sample.flatten()

                signal_to_noise = (1./noise_scale) * np.dot(weight_image,signal_image) \
                        / np.sum(weight_image)
                
                ## Make folders for storage
                f_path = 'Plots/' + str("%.3e" % (noise_scale)) + '/'
                mkdir_p(f_path)

                k=0
                for i in xrange(noise_img_num):
                                
                    ## Add the noise_matrix to the 0 noise image
                    image = image_0 + noise_scale*noise_img[:,i].reshape(size_X,size_Y)
                    
                    image_reconst_curr, coeffs_curr =\
                            shapelet_decomposition(image_data,\
                            f_path = f_path,\
                            N1=N1,N2=N2,basis=basis,solver=solver,\
                            image=image, coeff_0=coeff_0, noise_scale=signal_to_noise,\
                            n_max = n_max,\
                            plot_decomp = True,\
                            column_number = i)

                    if coeffs_curr != None:
                        coeff_stability[:,k] = coeffs_curr
                        k+=1

                plot_stability(coeff_stability, coeff_0, N1, N2, noise_img_num, \
                        n_max = n_max, label_arr = label_arr, \
                        beta_array = beta_array, signal_to_noise = signal_to_noise,\
                        basis = basis, solver = solver, \
                        fname_imag = f_path + 'Stability/',\
                        variance_folder = f_path + 'Stability/',\
                        mid_word = str(sum_max_order(basis,get_max_order(basis,n_max)))+'_'+str(basis))
         
if __name__=='__main__':
    
    Num_of_shapelets_array = [10,10,21]
    methods = ['lasso', 'sparse', 'lstsq']
    
    ## Range chose so that SNR is in range ~20 -- ~50
    noise_array = np.logspace(1.1, 1.5, 5)
    alpha_ = np.logspace(-5,-1.3,6)
    basis_array = ['Elliptical', 'Polar', 'XY', 'Compound']
    

    # Generate noisy images
    # galsim images are 78 x 78
    size_X = 78; size_Y=78
    noisy_mat_num = 10

    noise_matrix = np.zeros((size_X*size_Y , noisy_mat_num))
    for i in xrange(noisy_mat_num):
        noise_matrix[:,i] = np.random.randn(size_X*size_Y)

    for noise_scale in noise_array:
        # Select a method for fitting the coefficients
        for basis in basis_array[3:]:
            
            for solver in ['sparse']:#range(len(methods)): 

                test_stability(solver, basis, \
                        noise_scale, noise_img = noise_matrix, noise_img_num = noisy_mat_num,\
                        size_X = size_X, size_Y = size_Y,\
                        alpha_ = alpha_, Num_of_shapelets_array = Num_of_shapelets_array)
           
    #show_some_shapelets()
    #p_shapelet.plot_shapelets(6,2,1)
    #check_orthonormality()
