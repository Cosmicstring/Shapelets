import numpy as np
import numpy.linalg as linalg

from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from sklearn import linear_model

from plotting_routines import plot_solution
from utils.I_O_utils import *
from utils.shapelet_utils import *

#import pdb; pdb.set_trace()

## Flag for comparison with the omp, probably not using
## it in any pipeline computation

flag_compare = False


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

def solver_omp(D, signal, N1,N2, Num_of_shapelets = None):

    """ Find appropriate weights for the basis coefficients 
    obtained by the inner product routine in __shapelet_decomposition__
    using the Orthogonal Matching Pursuit algorithm

    D       : ndarray containing the dictionary of shapelet basis
    signal  : ndarray original image to be decomposed into shapelet basis
    N!,N2   : Integer numbers of n and m quantum numbers respectively
    
    """
    n_nonzero_coefs = Num_of_shapelets
    if Num_of_shapelets == None:
        n_nonzero_coefs = N1*N2/4
    omp = OMP(n_nonzero_coefs = n_nonzero_coefs)
    omp.fit(D,signal)
    sparse_coefs = omp.coef_
    sparse_idx = sparse_coefs.nonzero()
    sparse_reconst = np.dot(D,sparse_coefs)
    sparse_residual = signal - sparse_reconst

    residual_energy_fraction = np.sum(sparse_residual**2)/np.sum(signal**2)
    recovered_energy_fraction = np.sum(sparse_reconst**2)/np.sum(signal**2)

    return sparse_coefs, sparse_reconst, sparse_residual, \
            residual_energy_fraction, recovered_energy_fraction, n_nonzero_coefs

def decide_nonzero(sigma_arr, max_sigma, take_nonzero, epsilon):

    """
    Returns the least conservative number of shapelets
    for the svd cut
    """

    n_nonzero = 0

    clip_sigma = max_sigma * epsilon
    ## Look how many values get discarded by epsilon cut
    len_clip_sigma = len(np.where(sigma_arr >= clip_sigma)[0])

    if (len_clip_sigma > take_nonzero):
        ## Return the values which are not less than clip_sigma
        return sigma_arr[np.where(sigma_arr > clip_sigma)]
    else:
        ## Otherwise if I throw out less of the coefs by take_nonzero
        ## then just return first nonzero // biggest nonzero
        return sigma_arr[:take_nonzero]

def solver_SVD(D, signal, take_nonzero = 21, epsilon = 0.01, decomp_method = 'Dual'):

    """ Find appropriate coefficients for basis vectors contained in D, reconstruct image,
    calculate residual and residual and energy fraction using the Singular Value Decomposition
    
    D       : ndarray of dictionary for shapelet basis
    signal  : ndarray representing the original image

    """

    rows_SVD, columns_SVD = np.shape(D)
    U, s0, VT = linalg.svd(D, full_matrices = True)    
  
    ## Count in only n_nonzero signular values set rest to zero
    ## it seems the No of dominant ones is the same as the No of shapelets
    ## in the basis    
    ## Filter out corresponding columns in VT / rows in U / singular values in s
    s = decide_nonzero(s0, max(s0), take_nonzero, epsilon)
    n_nonzero = len(s)

    print 'Biggest trashed singular value: ', s0[n_nonzero]
    
    ## Rows of V span the row space of D
    #VT[:,n_nonzero:] = 0
    ## Columns of U span the column space of D
    #U[:,n_nonzero:] = 0
    
    ## In the docs it is said that the matrix returns V_transpose and not V 
    V = VT.transpose() 
    ## Initialize diagonal matrices for singular values
    S = np.zeros_like(D)
    S_dual = np.zeros_like(D)

    ## Put singular values on the diagonal
    for i in xrange(n_nonzero):
        S[i,i] = s[i]
        S_dual[i,i] = 1./s[i]
 
    if decomp_method == 'Dual':
        coeffs_SVD_r = np.dot(V, np.dot(S_dual.transpose(), np.dot(U.transpose(),signal))) 
        ## If it is prefered to select top n_nonzero coefs
        if n_nonzero < len(np.where(coeffs_SVD_r != 0)[0]):
            sorted_indices = np.abs(coeffs_SVD_r).argsort()
            print "Dual"
            print "Biggest thrown out coeff: ", coeffs_SVD_r[sorted_indices[-(n_nonzero+1)]]
            idx_nonzero = sorted_indices[-n_nonzero:]
            for i in xrange(len(coeffs_SVD_r)):
                if not(i in idx_nonzero):
                    coeffs_SVD_r[i] = 0

    elif decomp_method == 'Pseudo_Inverse':
        ## Initialize the coeffs array
        
        coeffs_SVD_r = np.zeros(D.shape[1])
        
        ## Shrink the matrix and select only the nonzero columns
        ## the zero columns are the consequence of the construction
        ## of the basis matrix
        
        idx_col = []
        for j in xrange(D.shape[1]):
            ## If the column has a shapelet vector inside
            ## take it
            if (len(np.where(D[:,j]!=0)[0]) != 0):
                idx_col.append(j)
        idx_col = np.asarray(idx_col)
        
        ## Make the coefficient vector with the pseudo inverse of 
        ## the SVD matrix
        
        E = D[:,idx_col]
        ET = E.transpose()
        
        Mat_sudo_inv = linalg.inv(np.dot(ET,E)) 
        coeffs_SVD_tmp = np.dot(Mat_sudo_inv,np.dot(ET,signal))
        
        ## Mat_sudo_inv is [n_nonzero X n_nonzero] so for the latter
        ## reconstruction this needs to be set
        ## Map the coefficients back to the corresponding indices
        ## of the resulting coefficient array
        
        i = 0
        for idx in idx_col:
            coeffs_SVD_r[idx] = coeffs_SVD_tmp[i]
            i+=1
        
        ## If n_nonzero is different than the basis size
        ## take n_nonzero biggest ones by abs value
        if n_nonzero < len(idx_col):
            sorted_indices = np.abs(coeffs_SVD_r).argsort()
            idx_nonzero = sorted_indices[-n_nonzero:]
            print "Pseudo Inverse"
            print "Biggest thrown out coeff: ", coeffs_SVD_r[sorted_indices[-(n_nonzero+1)]]
            for i in xrange(len(coeffs_SVD_r)):
                if not(i in idx_nonzero):
                    coeffs_SVD_r[i] = 0.

    n_nonzero_coefs_SVD = np.count_nonzero(coeffs_SVD_r)
    reconstruction_SVD = np.dot(D,coeffs_SVD_r)
    residual_SVD = signal - reconstruction_SVD
    residual_energy_fraction_SVD = np.sum(residual_SVD**2)/np.sum(signal**2)
    recovered_energy_fraction_SVD = np.sum(reconstruction_SVD**2)/np.sum(signal**2)
    
    return coeffs_SVD_r, reconstruction_SVD, residual_SVD, \
            residual_energy_fraction_SVD, recovered_energy_fraction_SVD, n_nonzero_coefs_SVD

def solver_lstsq(D, signal):

    """Find appropriate coefficients for the basis matrix D, reconstruct the image, calculate
    residual and energy and residual fraction using the Orthogonal Matching Pursuit algorithm
    
    D       : ndarray, dictionary of shapelet basis
    signal  : ndarray of the original image
    """
    
    coeffs_lstsq = linalg.lstsq(D, signal)[0]  
    n_nonzero_coefs_lstsq = np.count_nonzero(coeffs_lstsq) 
    reconstruction_lstsq = np.dot(D,coeffs_lstsq)
    residual_lstsq = signal - reconstruction_lstsq
    residual_energy_fraction_lstsq = np.sum(residual_lstsq**2)/np.sum(signal**2)
    recovered_energy_fraction_lstsq = np.sum(reconstruction_lstsq**2)/np.sum(signal**2)
    
    return coeffs_lstsq, reconstruction_lstsq, residual_lstsq, \
            residual_energy_fraction_lstsq, recovered_energy_fraction_lstsq, n_nonzero_coefs_lstsq

def solver_lasso_reg(D, n_nonzero, signal, alpha = None):
    
    """Find appropriate basis coefficients for the vectors inside basis matrix D
    reconstruct the image, calculate residual and energy and residual fraction with 
    the Lasso regularization technique minimizing the L_1 norm
    
    D       : ndarray dictionary of shapelet basis
    signal  : ndarray of the original image for decomposition

    """
    if (alpha == None):
        alpha = 0.001
    
    lasso_fit = linear_model.Lasso(alpha = alpha, max_iter=10000, fit_intercept = False).fit(D, signal)
    coeffs_lasso = lasso_fit.coef_
   
    ## Just for comparison of top 21 with the omp
    ## Don't do this otherwise
    if flag_compare:
        coeffs_lasso1 = np.zeros_like(coeffs_lasso)
        idx_n_nonzero = np.abs(coeffs_lasso).argsort()[-n_nonzero:][::-1]
        coeffs_lasso1[idx_n_nonzero] = coeffs_lasso[idx_n_nonzero]
        coeffs_lasso = coeffs_lasso1

    reconstruction_lasso = np.dot(D, coeffs_lasso)
    
    residual_lasso = signal - reconstruction_lasso
    residual_energy_fraction_lasso = np.sum(residual_lasso**2)/np.sum(signal**2)
    recovered_energy_fraction_lasso = np.sum(reconstruction_lasso**2)/np.sum(signal**2)
    n_nonzero_coefs_lasso = np.count_nonzero(coeffs_lasso)

    return coeffs_lasso, reconstruction_lasso, residual_lasso, \
            residual_energy_fraction_lasso, recovered_energy_fraction_lasso, n_nonzero_coefs_lasso

def select_solver_do_fitting_plot(\
        f_path, \
        basis, \
        N1, N2, n_max, column_number, \
        image_initial, D, signal,solver, \
        beta_array,\
        noise_scale=0., coeff_0 = [], Num_of_shapelets = 10, alpha_ = 0.0001, \
        flag_gaussian_fit = True, plot = True):

    """
    As the name says, here the solver is selected, for weightning certain shapelets in the dictionary,
    by appropriate optimizing function. After the coefficients are obtained call the the plot_solution routine. 
    Currently enabled solvers are:
        -- Lasso regularization (solver_lasso)
        -- Orthogonal Matching Pursuit (solver_omp)
        -- Singular value decomposition (solver_SVD)
        -- Least squares (solver_lstsq)

    Input parameters:
    -----------------
    
    f_path          :   String representing the path to which all the plots in the plot_solution routine and intermediate outputs are saved
    basis           :   String representing the basis in which decomposition is done.
    N1,N2           :   Integer numbers representing N and M quantum numbers
    n_max           :   Integer defining the number of shapelets in one-beta-scale dictionary 
    image_initial   :   ndarray of the initial image which is decomposed into shapelet basis
    D               :   ndarray of the dictionary for shapelet basis
    signal          :   1D array, which is image_initial.flatten()
    solver          :   String selecting the solver for decomposition
    beta_array      :   1D array of beta scale values // Usually [sigma/4., sigma/2., sigma, sigma*2 sigma*4]

    Used only for stability tests:
    --------------------------------

    column_number   :   Integer representing the idx of the random gausian noise matrix
    noise_scale     :   Float number which defines the variance of the noise matrix added to initial image
    coeff_0         :   ndarray of the coefficients obtained for 0 noise_scale

    """

    ## Default for foler path
    folder_path_word = ""
    mid_path_word = ""
    end_word = ""
    if noise_scale != None:
        mid_name_word = '_solution_'+str("%.3e" % (noise_scale))+'_'\
            +str(N1)+'_'+str(N2)\
            +'_'+str(n_max)+'_'+basis+'_'+ str(column_number) + '_'
    else:
        mid_name_word = '_solution_'\
            +str(N1)+'_'+str(N2)\
            +'_'+str(n_max)+'_'+basis+ '_'

    ## Sparse solver
    if (solver == 'omp'):
        
        ## Include symmetric number of shapelets
        ## symmetrize the number chosen
        Num_of_shapelets = sum_max_order(basis,(get_max_order(basis,Num_of_shapelets))) 

        coeffs, reconst, residual, \
            residual_energy_fraction, recovered_energy_fraction, \
            n_nonzero_coefs = solver_omp(D, signal, N1,N2,Num_of_shapelets)
         
        mid_path_word = str(n_nonzero_coefs) + '_' + basis
        end_word = str(n_nonzero_coefs)
    ## SVD solver // following berry approach
    elif (solver == 'svd'):
       
        decomp_method = 'Dual'
        coeffs, reconst, residual, \
        residual_energy_fraction, recovered_energy_fraction, \
        n_nonzero_coefs = solver_SVD(D,signal, \
        decomp_method = decomp_method, take_nonzero = Num_of_shapelets, epsilon=0.01) 
        
        mid_path_word = decomp_method + '_' + str(n_nonzero_coefs) + '_' + basis

    ## Ordinary least squares solver
    elif (solver == 'lstsq'):  

        coeffs, reconst, residual, \
        residual_energy_fraction, recovered_energy_fraction, \
        n_nonzero_coefs = solver_lstsq(D, signal) 

        mid_path_word = str(n_nonzero_coefs) + '_' + basis
    
    elif (solver == 'lasso'): #This is with the Lasso regularization
           
        coeffs, reconst, residual, \
            residual_energy_fraction, recovered_energy_fraction, \
            n_nonzero_coefs = solver_lasso_reg(D, Num_of_shapelets,signal, alpha_)
    
        mid_path_word = str("%.3e" % (alpha_)) + '_'+ str(n_nonzero_coefs) + '_' + basis         

    
    end_word = str(n_nonzero_coefs)
    folder_path_word = f_path + solver + '/' + mid_path_word + '/'
 
    if noise_scale==None:
        coefs_plot = coeffs
    else:
        coefs_plot = asses_diff(coeffs,coeff_0)

    ## Make a dir for storage of decompositions 
    mkdir_p(folder_path_word)

    ## size_X and size_Y should be the size of the initial image
    size_X = image_initial.shape[0]
    size_Y = image_initial.shape[1]
    if plot == True:
        plot_solution(basis, N1,N2,image_initial,size_X,size_Y, \
            reconst, residual, coefs_plot,\
            recovered_energy_fraction, residual_energy_fraction, n_nonzero_coefs, \
            noise_scale, \
            folder_path_word\
            + solver + mid_name_word + end_word,\
            beta_array = beta_array,\
            flag_gaussian_fit = flag_gaussian_fit)
        
    return reconst.reshape(size_X,size_Y),coeffs
