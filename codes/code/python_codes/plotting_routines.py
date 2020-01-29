import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from utils.I_O_utils import *
from utils.shapelet_utils import *
import matplotlib.cm as cm

import os
"""
----------------------
massey_refregier_2005 == Massey R., Refregier A., MNRAS 2005
----------------------
"""

## Set up global LaTeX parsing
plt.rc('text', usetex=True)
plt.rc('font', **{'family' : "sans-serif"})
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', \
                r'\usepackage[utf8]{inputenc}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

#import pdb; pdb.set_trace()

def coeff_plot2d(coeffs,N1,N2,\
        ax=None,fig=None,\
        orientation='vertical',\
        f_coef_output = ''):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    """
    Used for plotting the Cartesian basis coefficients. More convinient to plot them
    in matrix form than Polar basis. Figure and Axes objects are passed from another plotting
    routine and returned changed.

    Input parameters:
    -----------------
    
    coeffs          :   ndarray of dimension N1xN2 of float coefficient values; if not of N1xN2 shape
                        then coeffs are reshaped to N1xN2 shape
    N1,N2           :   Integers defining the shape of coeffs
    ax,fig          :   Axes and Figure object passed by other plotting routine
    orientation     :   String defining the label orientation on the Axes
    f_coef_output   ;   String contatinig the path for .txt files // Optional, used only for debug
    

    Returns:
    --------
    ax,fig          :   Axes and Figure objects, with the coefficient plot
    """

    ## Double check write out to file
    ## coeffs

    if f_coef_output != '':
        f = open(f_coef_output, "w")
        f.write("N_1\tN_2\tCoeff values\n")

    if ax is None:
      fig, ax = plt.subplots()

    coeffs_reshaped = None

    if (coeffs.shape != (N1,N2)):
        coeffs_reshaped = coeffs.reshape(N1,N2) 
    else:
        coeffs_reshaped = coeffs
    
    idx_nonzero = np.where(coeffs_reshaped != 0)

    ## Output to the file only nonzero values of coefs
    if f_coef_output != '':
        for n,m in zip(idx_nonzero[0], idx_nonzero[1]):
            f.write("%d\t%d\t%.3e\n" % (n,m,coeffs_reshaped[n,m]))
        f.close()

    #coeffs_reshaped /= coeffs_reshaped.max()
    im = ax.imshow(coeffs_reshaped,cmap=cm.bwr, interpolation='none')
    
    #set yticks and xticks

    yticks_int = []; xticks_int = []
    for n,m in zip(idx_nonzero[0], idx_nonzero[1]):
        yticks_int.append(int(n)); xticks_int.append(int(m))

    # ax.set_yticks = yticks_int
    # ax.set_xticks = xticks_int

    ## Force colorbars besides the axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%')    

    fig.colorbar(im,cax=cax,format = '%.2e', orientation=orientation)
    return fig,ax

def coeff_plot_polar(coeffs, N1,N2, \
        len_beta = 1, N_range = 10,ax = None, fig = None, colormap = cm.bwr,\
        orientation = 'vertical',\
        f_coef_output = ''):
    """
    Plot the values of coeffs to the fig and ax objects obtained in the triangular grid as in massey_refregier_2005.
    Return fig and ax changed by this function.

    Return:
    -------
    ax, fig     :   Axes and Figure objects with plotted coefficients
    """
    import matplotlib as mpl
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ## For double check / if color scale is not too good
    ## write out the values of 
    ## the coefs
    if f_coef_output != '':
        f = open(f_coef_output, "w")
        f.write("N\tM\tCoeff value\n")
    ## Initialize the array for the colors / intenzities of the coefs
    color_vals = []

    ## For now 15 is the selected N_range
    x = []
    y = []
    
    ## Check if order of n in basis (N1 currently) is smaller than N_range
    if N1 < N_range:
        N_range = N1
    
    k = 0
    yticks_int = []
    xticks_int = []
    for n in range(N_range):
        for m in range(-n,n+1,2):
            x.append(n)
            y.append(m)
            color_vals.append(coeffs[k])
            if (coeffs[k] != 0):
                yticks_int.append(int(m))
                xticks_int.append(int(n))
                if f_coef_output !='':
                    f.write("%d\t%d\t%.3e\n" % (n,m, coeffs[k]))
            k += 1
        ## Make appropriate y coord
        ## So that the squares don't overlap
        ## and there is no white space between
        #y.append(np.linspace(-n,n+1,2))

    if f_coef_output !='':
        f.close()
    ## Merge everything into one array of the same shape as x
    x = np.asarray(x)
    y = np.asarray(y)    
    color_vals = np.asarray(color_vals)
    ## Control the size of squares
    dx = [x[1]-x[0]]*len(x) 
    
    ## Get the range for the colorbar
    norm = mpl.colors.Normalize(\
        vmin=np.min(color_vals),
        vmax=np.max(color_vals))

    ## create a ScalarMappable and initialize a data structure
    s_m = cm.ScalarMappable(cmap=colormap, norm=norm); s_m.set_array([])

    if fig == None:
        fig, ax = plt.subplots()
    
    ax.set_ylabel('m')
    ax.set_xlabel('n')
    ax.set_aspect(aspect = 'auto', adjustable ='box')
    xmin, xmax, ymin, ymax = ax.axis([min(x)-1., max(x)+1., min(y)-1., max(y)+1.])
    #ax.minorticks_on()

    # Set the yticks and xticks to be integers

    ax.set_yticks(list(yticks_int))
    ax.set_xticks(list(xticks_int))

    # Mark the position where there are non-zero valued coefficients

    ax.plot(xticks_int, yticks_int, marker = 'x', color = 'black')


    ## Add the coeffs as squares
    ## without any white spaces remaining
    for x,y,c,h in zip(x,y,color_vals,dx):
        ax.add_artist(\
                Rectangle(xy=(x-h/2., y-h),\
                linewidth=1, color = s_m.to_rgba(c),\
                width=h, height=2*h))
    
    ## Locate the axes for colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%")

    fig.colorbar(s_m, format = '%.2e', cax = cax)
    
    return fig, ax

def _get_gaussian_noise_params(arr,f_path, bins_num = None):
    """
    Make a histogram distribution of the values in arr and return back parameters
    of the best fit gaussian to that (it is assumed that the distr. is gaussian).
    As a side product plot the obtained gaussian distribution to the f_path.

    Parameters:
    -----------

    arr     : 1D array of input values for which histogram is going to be calculated
    f_path  : String path variable controling the path where the resulting plot is going 
             to be saved

    Optional:
    ---------

    bins_num        : Integer number N+1 of bins to be provided in the histogram
    initial_guess   : 1D array of initial guesses for the curve_fit function


    Returns:
    --------

    p_fit : 2 - length array with obtained sigma and mu values from the fit
    p_err : 2-length array containing std of sigma and mu
    """

    from scipy.optimize import leastsq, curve_fit
    from scipy.stats import norm

    ## Make a callable fucntion gaussian
    gaussian_lstsq = lambda p, x: \
            1./np.sqrt(2*np.pi*p[0]**2) * np.exp(- (x - p[1])**2 / (2.* p[0]**2))

    gaussian = lambda x, a, b, mu, sigma: \
            a * np.exp(- (x - mu)**2 / (2.* sigma**2)) + b
    
    errfunc = lambda p, x,y: (y - gaussian_lstsq(p,x))

    if (bins_num == None):
        bins_num = int(np.floor(0.02*len(arr)))
    
    hist, bin_edges_tmp  = np.histogram(arr, bins = bins_num, density = True)

    ## Flatten the bin_edges array by takin the central values
    bin_edges = np.zeros(len(hist))

    for i in range(len(hist)):
        bin_edges[i] = (bin_edges_tmp[i] + bin_edges_tmp[i+1])/2.

    ## Get the initial guess
    mu_init, sigma_init = norm.fit(arr)
    
    ## Do the fit
    initial_guess = [sigma_init, mu_init]
    p_fit, p_cov, ifodict, errmsg, success = \
            leastsq(errfunc, initial_guess, args = (bin_edges, hist), full_output = True)
    ## If fit is success then continue, otherwise return
    ## None
    if success in [1,2,3,4]:
            
        ## Erros hav to be calculated manualy
        s_sq = (errfunc(p_fit, bin_edges, hist)**2).sum()/(len(hist) - len(initial_guess))
        p_cov = p_cov * s_sq
        
        p_err = np.sqrt(np.abs(np.diag(p_cov)))
        
        fig, ax = plt.subplots()
        ax.plot(\
                bin_edges, hist, 'bo', label ='data')
        ax.plot(\
                bin_edges, gaussian_lstsq(p_fit, bin_edges), 'r-', label = 'fit')
        ax.set_ylabel('PDF of pixel values')
        ax.set_xlabel('Pixel values')
        ax.legend(loc='best')
        if not(os.path.exists(f_path)):
            plt.savefig(f_path)
        plt.clf()
        plt.close()

        return p_fit[0], p_fit[1], p_err[0], p_err[1]
    else:
        return None, None, None, None

def plot_decomposition(basis, image, size_X, size_Y, \
        base_coefs,N1,N2,shapelet_reconst_array, signal, residual_array,\
        residual_energy_fraction_array,recovered_energy_fraction_array, Path,\
        beta_array = []):

    """ Plot the decomposition obtained with the chosen solver and save the result to Path

    Parameters:
    -----------
    basis                       : String variable which controls the selected __basis__ in which decomposition was made
    image                       : Ndarray representing image that was provided for decomposition
    size_X, size_Y              : Integer numbers defining image X and Y sizes
    base_coefs                  : 1D array of float base coefficients obtained from the decomposition
    N1,N2                       : Integer numbers representingi max N and M qunatum numbers respectively
    shapelet_reconst            : 1D array of reconstruction of the image with the obtained base_coefs
    signal                      : 1D array of an image vector, obtained from flattening the original image matrix
    residual                    : 1D array of residual obtained with difference between signal and shapelet_reconst
    residual_energy_fraction    : 1D array energy fraction of the residual image
    recovered_energy_fraction   : 1D array energy fraction of the obtained image with shapelet_reconst
    beta_array                  : 1D array with betas used // Added for the compound basis

    """ 
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if basis =='Compound_XY':
        base_coefs = base_coefs.reshape(N1,N2*len(beta_array))    
    elif basis == 'XY' or basis == 'XY_Elliptical':
        base_coefs = base_coefs.reshape(N1,N2)

    for i in range(len(beta_array)):
       
        shapelet_reconst = shapelet_reconst_array[i]
        residual = residual_array[i]
        residual_energy_fraction  = residual_energy_fraction_array[i]
        recovered_energy_fraction = recovered_energy_fraction_array[i]

        str_beta = str("%.3f" % (beta_array[i]))

        if ('XY' in basis):
            left_N2 = i*N2
            right_N2 = (i+1)*N2
            coefs = base_coefs[:N1, left_N2:right_N2]
        else:
            left = i*(N1*N2)
            right = (i+1)*(N1*N2)
            coefs = base_coefs[left:right]
            
        if np.count_nonzero(coefs) != 0:

            print('beta ', beta_array[i], 'Decomp ', coefs.shape)

            fig, ax = plt.subplots(2,2, figsize = (10, 10))
            if ('XY' in basis):
                foo, ax[1,1] = coeff_plot2d(coefs,N1,N2,\
                        ax=ax[1,1],fig=fig,\
                        f_coef_output = Path + '_' + str_beta + '_.txt')
            elif ('Polar' in basis):
                foo, ax[1,1] = coeff_plot_polar(coefs,N1,N2,\
                        ax=ax[1,1], fig=fig,
                        f_coef_output = Path + '_' + str_beta + '_.txt')

            vmin, vmax = min(shapelet_reconst.min(),signal.min()), \
                    max(shapelet_reconst.max(),signal.max())

            im00 = ax[0,0].imshow(image,
                                    vmin=vmin,vmax=vmax,cmap = cm.jet)
            im01 = ax[0,1].imshow(shapelet_reconst.reshape(size_X,size_Y),
                                    vmin=vmin,vmax=vmax, cmap = cm.jet)
            im10 = ax[1,0].imshow(residual.reshape(size_X,size_Y),
                                    cmap = cm.jet)

            # Force the colorbar to be the same size as the axes
            divider00 = make_axes_locatable(ax[0,0])
            cax00 = divider00.append_axes("right", size="5%")

            divider01 = make_axes_locatable(ax[0,1])
            cax01 = divider01.append_axes("right", size="5%")

            divider10 = make_axes_locatable(ax[1,0])
            cax10 = divider10.append_axes("right", size="5%")

            fig.colorbar(im00,format = '%.2e', cax=cax00); 
            fig.colorbar(im01,format = '%.2e', cax=cax01); 
            fig.colorbar(im10,format = '%.2e', cax=cax10)

            ax[0,0].set_title('Original image')
            ax[0,1].set_title('Reconstructed image - Frac. of energy = '\
                    +str(np.round(recovered_energy_fraction,4)))
            ax[1,0].set_title(\
                    'Residual image - Frac. of energy = '\
                    +str(np.round(residual_energy_fraction,4)))
            
            ax[1,1].grid(lw = 2, which = 'both')
            ax[1,1].set_title('Values of coefficients')
            fig.suptitle('Shapelet Basis decomposition')
            
            fig.tight_layout()
            
            if not(os.path.exists(Path + '_' + str_beta + '_.png')):
                plt.savefig(Path + '_' + str_beta + '_.png')

            plt.clf()
            plt.close()

def plot_solution(basis, N1,N2,image_initial,size_X, size_Y,\
        reconst, residual, coefs_initial,\
        recovered_energy_fraction, residual_energy_fraction, \
        n_nonzero_coefs, noise_scale, Path,\
        beta_array = [-1.],\
        flag_gaussian_fit = True,\
        title_00 = '', title_01 = '', title_10 = '', title_11 = ''):

    """ Plot obtained images from the coefficients obtained with the selected __solver__
    
    Parameters:
    -----------
    basis                       : String variable which controls the basis for decomposition
    N1,N2                       : Integer numbers of max N and M quantum numbers
    image_initial               : 2Darray of initial image provided for decomposition
    size_X,size_Y               : Float numbers of X and Y size of the image_initial
    reconst                     : 1D array of floats, representing reconstruction of the image obtained
    coeffs_initial              : 1D array of coeffs obtained in the reconstruction process
    residual                    : 1D array of residual obtained with difference between signal and shapelet
    residual_energy_fraction    : 1D array og energy fraction of the residual image
    recovered_energy_fraction   : 1D array of energy fraction of the obtained image with shapelet_reconst
    n_nonzero_coefs             : Integer defining nonzero coefficients in the coefs variable
    fig                         : Figure object forwarded from plot_decomposition
    Path                        : String representing the path where the final plot is saved
    
    Optional:
    ---------
    beta_array : 1D array of beta values used for basis matrix  

    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    flag = 0
    ## Get the residual gaussian fit // noise I generate is gaussian        
    if flag_gaussian_fit:
        sigma_res, mu_res, err_sigma_res, err_mu_res = \
            _get_gaussian_noise_params(residual,\
            Path + '_gaussian_noise_fit_.png')
    
        ## If fit failed set flag to 0
        if (sigma_res == None):
            flag = 0
        else:
            flag = 1 

    for i in range(len(beta_array)): 
        
        str_beta = str("%.3f" % (beta_array[i]))

        if ('XY' in basis):
            left = i*(N1*N2)
            right = (i+1)*(N1*N2)
            coefs = coefs_initial[left:right].reshape(N1,N2)
        else:
            left = i*(N1*N2)
            right = (i+1)*(N1*N2) 
            coefs = coefs_initial[left:right]
        
        if np.count_nonzero(coefs) != 0:
            print('beta ', beta_array[i], 'shape: ', coefs.shape)


            fig2, ax2 = plt.subplots(2,2, figsize = (10,10))
            vmin, vmax = min(reconst.min(),image_initial.min()), \
                    max(reconst.max(),image_initial.max())
            
            im00 = ax2[0,0].imshow(image_initial, 
                                    aspect = '1', vmin=vmin, vmax=vmax, cmap = cm.jet)
            im01 = ax2[0,1].imshow(reconst.reshape(size_X,size_Y), 
                                    aspect = '1', vmin=vmin, vmax=vmax, cmap = cm.jet)
            im10 = ax2[1,0].imshow(residual.reshape(size_X,size_Y), 
                                    aspect = '1', cmap = cm.jet)

            if 'XY' in basis:
                fig2, ax2[1,1] = coeff_plot2d(coefs,N1,N2,\
                        ax=ax2[1,1],fig=fig2,\
                        f_coef_output = Path + '_' + str_beta + '_.txt') 
            elif 'Polar' in basis:
                fig2, ax2[1,1] = coeff_plot_polar(coefs,N1,N2,\
                        ax=ax2[1,1],fig=fig2,\
                        f_coef_output = Path + '_' + str_beta + '_.txt')     
                ax2[1,1].grid(lw=2)
            
            # Force the colorbar to be the same size as the axes
            divider00 = make_axes_locatable(ax2[0,0])
            cax00 = divider00.append_axes("right", size="5%")

            divider01 = make_axes_locatable(ax2[0,1])
            cax01 = divider01.append_axes("right", size="5%")

            divider10 = make_axes_locatable(ax2[1,0])
            cax10 = divider10.append_axes("right", size="5%")  
            
            fig2.colorbar(im00, format = '%.2e', cax=cax00)
            fig2.colorbar(im01, format = '%.2e', cax=cax01) 
            fig2.colorbar(im10, format = '%.2e', cax=cax10)
            ax2[0,0].set_title('Original image'); 
            ax2[0,1].set_title('Reconstructed image - Frac. of energy = '\
                    +str(np.round(recovered_energy_fraction,4)))
                                
            if flag:

                #exp_sigma = r'$\displaystyle \cdot 10^{-2}$'
                #token_sigma = "%.2e"
                #get_len_sigma = 4

                #str_err_sigma = str(token_sigma % (err_sigma_res))[:get_len_sigma]
                #str_sigma = str(token_sigma % (sigma_res))[:get_len_sigma]

                #exp_mu = r'$\displaystyle \cdot 10^{-2}$'
                #token_mu = "%.2e"
                #get_len_mu = 4

                #str_mu = str(token_mu % (mu_res))[:get_len_mu]
                #str_err_mu = str(token_mu % (err_mu_res))[:get_len_mu]

                ax2[1,0].set_title(\
                    ('Residual image - frac. of energy = '\
                    +str(np.round(residual_energy_fraction,4))\
                    + '\n' \
                    + r'$\displaystyle \sigma = $'\
                    + '(%.2e' + r'$\pm$' + '%.2e)' \
                    + '\n' \
                    + r'$\displaystyle \mu = $'\
                    + '(%.2e' + r'$\pm$' + '%.2e)') % \
                    (sigma_res, err_sigma_res, mu_res, err_mu_res))
            else:
                ax2[1,0].set_title(\
                    'Residual image - Frac. of energy = '\
                    +str(np.round(residual_energy_fraction,4)))
                    #+ '\n'\
                    #+ 'gauss fit failed')

            fig2.suptitle('Sparse decomposition from an semi-intelligent Dictionary')

            if (noise_scale == 0) or (noise_scale == None):
                ax2[1,1].set_title('Values of coefficients - '\
                        + str(n_nonzero_coefs) \
                        + '\n' \
                        + 'beta - ' + str_beta)
            else:
                ax2[1,1].set_title('Rel. diff in values ' \
                        + r'$\displaystyle \left|\frac{N.C_i}{O.C_i} - 1 \right|$'\
                        + ' - ' + str(n_nonzero_coefs) \
                        + '\n' \
                        + 'beta - ' + str_beta)
            
            fig2.tight_layout()
            if not(os.path.exists(Path + '_' + str_beta + '_.png')):
                plt.savefig(Path + '_' + str_beta + '_.png')
            plt.clf()
            plt.close()

def stability_plots(basis,solver,coefs,\
        N1,N2,\
        f_path_to_save,y_axis_scale = '',\
        ax_title = '', f_coef_output = ''):
    
    if not np.all(coefs==0):
        fig, ax = plt.subplots()
            
        if 'XY' in basis:
            fig,ax=coeff_plot2d(coefs,N1,N2,ax=ax,fig=fig,\
                    f_coef_output = f_coef_output) 
        elif 'Polar' in basis:
            fig,ax=coeff_plot_polar(coefs,N1,N2,ax=ax,fig=fig,\
                    f_coef_output = f_coef_output)
            ax.grid(lw=2, which='both')
        
        if y_axis_scale != '':
            ax.set_yscale(y_axis_scale)
        
        ax.set_title(ax_title)

        fig.tight_layout()
        if not(os.path.exists(f_path_to_save + '_.png')):
            plt.savefig(f_path_to_save + '_.png')

        return fig, ax
    else:
        print("All coefs zero for:\n")
        print("%s\n" % (f_path_to_save))
        return None, None

def plot_stability(coeff_stability, coeff_0, N1, N2, noise_img_num, \
        n_max = 20, label_arr = None,\
        beta_array = [], signal_to_noise = None, \
        basis = 'Polar', solver = 'lasso', \
        path_to_save = 'Plots/lasso/Stability/',\
        mid_word = '',flag_plot=1):
    
    """
    This plotting is used only for stability tests. It plots std matrix, relative std matrix, difference of coefficients and relative change of
    the coefficient value. All the plots are saved to the path_to_save + additional_path_string file path.
    """
    
    from matplotlib.ticker import FormatStrFormatter
    
    ## Initialize the folder for saving the images
    mkdir_p(path_to_save)

    
    ## Initialize the stability and std_rel arrays
    
    len_coeffs = coeff_stability.shape[0]
    coeff_stability_res = np.zeros(len_coeffs)
    coeff_mean_value = np.zeros(len_coeffs)
    coeff_diff = np.zeros(len_coeffs)
    std_rel_arr = np.zeros(len_coeffs)
    std_arr = np.zeros(len_coeffs)

    ## Take account for the rarely picked coeffs
    path_to_save_std_arr = path_to_save + 'std_data/'
    mkdir_p(path_to_save_std_arr)

    picked_rarely = open(path_to_save_std_arr +'rarely_picked.txt', 'w')
    f_var = open(path_to_save_std_arr + 'std_' + mid_word + '_.txt', 'w')
    f_var.write("Label\tVar\tMean val\n")

    idx_beta = 0

    ## Find the mean coefficients and the std_rel
    ## <N.C>_i and Var(N.C_i)
    for i in range(len_coeffs):
        
        ## Add all the values in different noise realizations
        ## of the same coordinate
        coeff_stability_res[i] = np.sum(coeff_stability[i,:])
        
        ## Calculate the std_rel of the i_th coordinate
        std_i = np.std(coeff_stability[i,:])
        std_arr[i] = std_i

        ## Calculate the mean of the i_th coordinate
        coeff_mean_value[i] = coeff_stability_res[i]/noise_img_num
        coeff_stability_res[i] = coeff_stability_res[i]/noise_img_num
        
        ## Calculate the difference
        coeff_diff[i] = coeff_stability_res[i] - coeff_0[i]
        
        if (i%(N1*N2) == 0):
            picked_rarely.write("Beta basis %.2e\n" % (beta_array[idx_beta]))
            idx_beta+=1

        ## Calculate the stability
        if coeff_0[i] !=0:
            coeff_stability_res[i] = np.abs(coeff_stability_res[i]/coeff_0[i] - 1)
            ## Store the variances for only corresponding
            ## coeff_0 coefficients, don't store the ones that are
            ## rarely picked
            f_var.write("%s\t%.10f\t%.10f\n" % \
                (label_arr[i], std_arr[i], coeff_mean_value[i])) 
        elif coeff_0[i]==0 and coeff_stability_res[i]!=0:
            picked_rarely.write("%s\t%.2e\n" % (label_arr[i], coeff_stability_res[i]))
            coeff_stability_res[i] = 0.
        
        ## Calculate relative magnitude of std 
        if coeff_mean_value[i] != 0:
            std_rel_arr[i] = std_i/np.abs(coeff_mean_value[i])
        else:
            std_rel_arr[i] = 0
    
    picked_rarely.close()
    f_var.close()

    if flag_plot == 1: 

        ## Calculate the relative std_rel of decompositions
        ## Coeff_stability is already in the needed form
        ## std_rel_i = Var(N.C_i)
        
        str_s_to_n = str("%.3e" % (signal_to_noise)) 

        for i in range(len(beta_array)):
            
            str_beta = str("%.2e" % (beta_array[i]))
    
            left = i*(N1*N2)
            right = (i+1)*(N1*N2)
            label_arr_curr = label_arr[left:right]
            
            std_rel_curr = std_rel_arr[left:right]
            std_arr_curr = std_arr[left:right]
            coefs = coeff_stability_res[left:right]
            coefs_diff = coeff_diff[left:right]
            coef_mean_curr = coeff_mean_value[left:right]
            coeff_0_curr = coeff_0[left:right]

            ## Add the scatter plot
            ## How many points to plot  
            n_max = sum_max_order(basis, get_max_order(basis,n_max))    
            n_nonzero = np.count_nonzero(coefs)
            n_nonzero = sum_max_order(basis,get_max_order(basis, n_nonzero))

            ## Take the biggest possible number of 
            ## coefficients
            if n_max <= n_nonzero:
                N_plot = n_max
            else:
                N_plot = n_nonzero

            N_plot = int(N_plot)
            ## If you can get all of the nonzero if not
            ## then just N_plot values
            ## If N_plot > nonzero * it returns the whole set of indices
            get_idx = np.abs(coeff_0_curr).argsort()[::-1][:N_plot]

            if n_nonzero != 0:
                coeff_mean_r = coef_mean_curr[get_idx]
                coeff_res_r = coefs[get_idx]
                std_arr_r = std_arr_curr[get_idx] 
                label_arr_r = label_arr_curr[get_idx]
                
                arange_x = np.arange(len(get_idx))
                
                fig_scat, ax_scat = plt.subplots(2, sharex=True)
 
                ## For some reason if this is grouped together
                ## it won't process the string correctly
                str_num_of_coefs = ' for %d biggest ' % (N_plot)
                title_0_string = \
                        'Scatter plot of '\
                        + r'$\displaystyle \left<N.C._i \right>$'\
                        + str_num_of_coefs\
                        + r'$\displaystyle \left<O.C._i \right>$'\
                        + ' coeffs'
                
                ax_scat[0].set_title(title_0_string)

                ax_scat[1].set_title(\
                        'Scatter plot of'\
                        + r'$\displaystyle \left|\frac{\left<N.C._i\right>}{O.C._i} - 1 \right|$')
                ax_scat[0].set_yscale('symlog')
                ax_scat[1].set_yscale('symlog')

                extraticks_0 = list(coeff_mean_r[np.abs(coeff_mean_r).argsort()[-2:][::-1]])
                extraticks_1 = list(coeff_res_r[np.abs(coeff_res_r).argsort()[-2:][::-1]])

                ax_scat[0].set_yticks(list(ax_scat[0].get_yticks()) + extraticks_0)
                ax_scat[1].set_yticks(list(ax_scat[1].get_yticks()) + extraticks_1)
                ax_scat[0].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
                ax_scat[1].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
                
                ax_scat[0].errorbar(arange_x, coeff_mean_r, yerr=std_arr_r, fmt='bo', \
                        label='Coeff. value')
                ax_scat[1].errorbar(arange_x, coeff_res_r, yerr=std_arr_r,fmt='ro',\
                        label='Coeff. stability')

                plt.xticks(arange_x, label_arr_r)
                ax_scat[1].set_xticklabels(\
                        ax_scat[1].get_xticklabels(),rotation=90, horizontalalignment = 'right')
                
                ax_scat[0].tick_params(axis='x', which ='both', pad = 10)
                ax_scat[1].tick_params(axis='x', which ='both', pad = 10)
                ax_scat[0].set_xlim(min(arange_x) - 1, max(arange_x) + 1)
                ax_scat[1].set_xlim(min(arange_x) - 1, max(arange_x) + 1)

                ## Set the font of the ticks
                for tick in ax_scat[0].xaxis.get_major_ticks():
                    tick.label.set_fontsize(7)
                for tick in ax_scat[0].xaxis.get_minor_ticks():
                    tick.label.set_fontsize(7)
                
                fig_scat.tight_layout()
                plt.savefig(path_to_save \
                        + solver + '_' + mid_word +'_'+str_beta+ "_scatter_coefs.png")
                plt.clf()
                plt.close()
                    
                ## Plot the stability of coeffs
                ax_title = 'Diff of coefs '\
                    + r'$\displaystyle \left<N.C._i\right> - O.C._i$' \
                    + '\n' \
                    + 'N.C is averaged over the number of noise realizations' \
                    + '\n' \
                    + 'S/N = ' + str_s_to_n\
                    + '\n'\
                    + 'beta - ' + str_beta
                
                f_path_to_save = path_to_save + solver + '_diff_'+mid_word+'_'+str_s_to_n+'_'\
                    +str_beta

                stability_plots(basis,solver,coefs_diff,\
                        N1,N2,\
                        f_path_to_save,\
                        ax_title = ax_title, f_coef_output = f_path_to_save + '_.txt')

                ## Plot the stability of coeffs
                ax_title = 'Stability of coefs '\
                    + r'$\displaystyle \left|\frac{<N.C.>_i}{O.C._i} - 1\right|$' \
                    + '\n' \
                    + 'N.C is averaged over the number of noise realizations' \
                    + '\n' \
                    + 'S/N = ' + str_s_to_n\
                    + '\n'\
                    + 'beta - ' + str_beta
                
                f_path_to_save = path_to_save + solver + '_stability_'+mid_word+'_'+str_s_to_n+'_'\
                    +str_beta

                stability_plots(basis,solver,coefs,\
                        N1,N2,\
                        f_path_to_save,\
                        ax_title = ax_title, f_coef_output = f_path_to_save + '_.txt')


                ## Plot relative std_rel
                ax_title = 'Rel. ' +r'$\displaystyle \sigma$'+ ' matrix '\
                        + r'$\displaystyle \sigma\left(N.C._i\right) / |\left< N.C._i\right>|$' \
                        + '\n' \
                        + 'S/N = ' + str_s_to_n\
                        + '\n'\
                        + 'beta - ' + str_beta
                
                f_path_to_save = path_to_save + solver + '_std_rel_'+mid_word+'_'+str_s_to_n+'_'\
                    +str_beta

                stability_plots(basis,solver,std_rel_curr,\
                        N1,N2,\
                        f_path_to_save,\
                        ax_title = ax_title, f_coef_output = f_path_to_save + '_.txt')

                ## Plot standard deviation

                ax_title = r'$\displaystyle \sigma$' ' matrix '\
                        + r'$\displaystyle \sigma\left(N.C._i\right)$' \
                        + '\n' \
                        + 'S/N = ' + str_s_to_n\
                        + '\n'\
                        + 'beta - ' + str_beta
                
                f_path_to_save = path_to_save + solver + '_std_' + mid_word + '_'\
                        + str_s_to_n+'_'\
                        +str_beta
                    
                stability_plots(basis,solver,std_arr_curr,\
                        N1,N2,\
                        f_path_to_save,\
                        ax_title = ax_title, f_coef_output = f_path_to_save + '_.txt')
                plt.close('all')
