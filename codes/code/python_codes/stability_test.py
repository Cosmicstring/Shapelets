import galsim
import time
import numpy as np

from astropy.io import fits

from shapelet_dicts import *
from plotting_routines import plot_stability

## Import the util package for the weights
from utils.galsim_utils import *
from utils.I_O_utils import *
from utils.shapelet_utils import *

# import pdb; pdb.set_trace()


def do_noise_iteration(image_0,image_data,noise_img,\
        size_X,size_Y,\
        N1,N2,\
        coeff_0,label_arr,beta_arr,\
        basis, solver,\
        noise_scale,noise_img_num,\
        Num_of_shapelets = 15, alpha = 0.0001,\
        plot_decomp = True, plot_sol = True,\
        n_max = 21, \
        mid_word = '', img_idx = 91, perturbed_flag=0):
    """
    Do the noise iterations and test the shapelet coefficients stability \
            and plot the result

    image_0         :   Noiseless image
    noise_img       :   Matrix with gaussian noise to be added to the image_0 \
                        with a factor of noise_scale
    size_X,size_Y   :   dimensions of image_0
    coeff_0         :   Shapelet coeffs obtained for the decomposition of the image_0
    noise_scale     :   factor with which noise_img is scaled
    noise_img_num   :   Number of noise_img matrices

    """
    str_perturbed_flag = ''
    if perturbed_flag:
        str_perturbed_flag = 'Beta_cluster_test/'
    ## The matrix to be used for stability calculations
    ## Variance and mean difference
    coeff_stability = np.zeros((len(coeff_0),noise_img_num))
    
    ## A sample image used for the S/N calculation
    image_sample = image_0 + noise_scale*noise_img[:,0].reshape(size_X,size_Y)
    
    ## Asses the weights and calculate S/N
    weight_image,flag = get_gaussian_weight_image(image_sample)
    
    if flag == 1:
        
        weight_image = weight_image.flatten()
        signal_image = image_sample.flatten()
        
        print("Testing the noise_scale generator")
        print(np.sum(weight_image**2) / 50.**2)

        signal_to_noise = (1./noise_scale) * np.dot(weight_image,signal_image) \
                / np.sum(weight_image)

        ## Make folders for storage
        f_path = 'Plots/'+ str("%d" % (img_idx)) + '/' + str("%.3e" % (noise_scale)) + '/'
        mkdir_p(f_path)

        f_SNR = open(f_path + "SNR_{}.txt".format(img_idx), "w+")

        f_SNR.write("Noise scale: {} \n".format(noise_scale))
        f_SNR.write("Signal to Noise: {} \n".format(signal_to_noise))
        f_SNR.close()

        k=0
        for i in range(noise_img_num):

            print("Noise iteration:\t%d" % (i))
            ## Add the noise_matrix to the 0 noise image
            image = image_0 + noise_scale*noise_img[:,i].reshape(size_X,size_Y)
 
            image_reconst_curr, coeffs_curr =\
                    shapelet_decomposition(image_data,\
                    f_path = f_path,\
                    N1=N1,N2=N2,basis=basis,solver=solver,\
                    image=image, coeff_0=coeff_0, noise_scale=signal_to_noise,\
                    Num_of_shapelets = Num_of_shapelets, alpha_ = alpha,\
                    plot_decomp = plot_decomp, plot_sol = plot_sol,\
                    beta_array = beta_arr,\
                    n_max = n_max,\
                    column_number = i)
            
            if np.all(coeffs_curr != None):
                coeff_stability[:,k] = coeffs_curr
                k+=1

        ## Temporary for control of plotting
        #if img_idx == 91:
        #    flag_plot = 1
        #else:
        #    flag_plot = 0

        plot_stability(coeff_stability, coeff_0, N1, N2, noise_img_num, \
                n_max = n_max, label_arr = label_arr, \
                beta_array = beta_arr, signal_to_noise = signal_to_noise, \
                basis = basis, solver = solver, \
                path_to_save = f_path + 'Stability_'+str_perturbed_flag+'/',\
                mid_word = mid_word, flag_plot=1)

def get_observed_image_decomp(\
        N1=20,N2=20,basis='XY',solver='omp',\
        image=None, coeff_0=None, noise_scale=0,\
        Num_of_shapelets = 21, alpha = 0.01, \
        plot_decomp = True,\
        select_img_idx = 91, n_max = 50):

    ## Take the same beta scale and theta as for the modeled image
    img_obs = fits.getdata('../../data/cube_real.fits')[select_img_idx]
    img_obs -= 1e6*0.16**2
    img_ = fits.getdata('../../data/cube_real_noiseless.fits')[select_img_idx]
    img_ -= 1e6*0.16**2

    galsim_image = galsim.Image(img_, scale = 1.0)
    shape = galsim_image.FindAdaptiveMom()
    
    image_data = np.zeros(5)
    x0,y0,sigma,beta,q = get_moments(shape)
    image_data[0] = x0; image_data[1] = y0; image_data[2] = sigma;
    image_data[3] = beta; image_data[4] = q

    if 'Compound' in basis:
        beta_array = np.asarray([sigma/4., sigma/2., sigma, sigma*2, sigma*4])
    else:
        beta_array = np.asarray([sigma])

    f_path = 'Stability_Plots/' + str(select_img_idx) + '/Observed/'
    mkdir_p(f_path)
 
    foo,foo =\
            shapelet_decomposition(image_data,\
            f_path = f_path,\
            N1=N1,N2=N2,basis=basis,solver=solver,\
            image=img_obs, coeff_0=None, noise_scale=-1.,\
            Num_of_shapelets=Num_of_shapelets, alpha_ = alpha, \
            plot_decomp = True, plot_sol=True,\
            make_labels=False,beta_array = beta_array,\
            n_max = n_max)

def prep_and_do_noise_iter(image_data,\
        noise_img,noise_img_num,noise_scale,\
        N1=20,N2=20,basis='XY',solver='omp',\
        image=None, coeff_0=None,\
        Num_of_shapelets = 21, alpha = 0.01, \
        plot_decomp = True,\
        select_img_idx = 51, n_max = 50, perturbed_flag=1):
    
    ## Make the no noise image
    f_path = 'Stability_Plots/' +str("%d" % (select_img_idx)) + '/' + str("%.3e" % (0.0)) + '/'
    mkdir_p(f_path)
    
    cube = fits.getdata('../../data/cube_real_noiseless.fits')
    background = 1.e6*0.16**2
    
    img = cube[select_img_idx] - background
    galsim_img = galsim.Image(img, scale = 1.0)
    
    shape_data = galsim_img.FindAdaptiveMom()
    
    ## Rescale pixels so that flux of image is 1.
    ## Useful for later clustering test
    img /= shape_data.moments_amp
    
    x0,y0,sigma,beta,q = get_moments(shape_data)
    image_data = [x0,y0,sigma,beta,q]

    if 'Compound' in basis:
        if not perturbed_flag:
            beta_array = np.asarray([sigma/4., sigma/2., sigma, sigma*2, sigma*4])
        else:
            # Here we want to also check how sensitive the reconstruction is
            # when we also give in the wrong beta scales for the image!
            beta_array = [0.990, 1.980, 3.960]
    else:
        beta_array = np.asarray([sigma])
    
    image_reconst, coeff_0, label_arr =\
            shapelet_decomposition(image_data,\
            f_path = f_path,\
            N1=N1,N2=N2,basis=basis,solver=solver,\
            image=img, coeff_0=None, noise_scale=0.,\
            Num_of_shapelets=Num_of_shapelets, alpha_ = alpha, \
            plot_decomp = plot_decomp, plot_sol = True, \
            make_labels = True,\
            n_max = n_max, beta_array = beta_array)
    
    if solver != 'lstsq':
        mid_word = str(\
                sum_max_order(basis,get_max_order(basis,Num_of_shapelets))) \
                + '_' + basis
    else:
        ## lstsq doesn't have any free params
        mid_word = basis

    ## If 0 noise decomp. fails don't do anything
    if (np.all(image_reconst != None)):
        do_noise_iteration(img,image_data,noise_img,\
                size_X,size_Y,\
                N1,N2,\
                coeff_0,label_arr,beta_array,\
                basis, solver,\
                noise_scale,noise_img_num,\
                Num_of_shapelets = Num_of_shapelets, alpha=alpha, \
                plot_decomp = False, plot_sol= False,\
                n_max = n_max,\
                mid_word = mid_word, img_idx = select_img_idx,\
                perturbed_flag = perturbed_flag)

def test_stability(solver, basis, \
        noise_scale, noise_img = None, noise_img_num = 10,\
        size_X = 78, size_Y = 78, \
        alpha_ = None, Num_of_shapelets_array = None, selected_images = [51]):
    
    """
    Do the stability test of different kinds of algorithms at disposal
        -- OMP
        -- SVD
        -- Lstsq
        -- Lasso
    """
    ## Initialize values
    N1 =20; N2=20; n_max = 50; Num_of_shapelets = None; alpha = None
    
    image = None; image_curr = None; coeffs_0 = None; coeffs_curr = None; coeff_stability = None
    
    ## Store x0,y0 (centroids) of the image and sigma
    image_data = np.zeros(5)

    ## Now select alphas if the method is lasso
    if solver == 'lasso':
        for alpha in alpha_:
            
            for Num_of_shapelets in [Num_of_shapelets_array[2]]:
            
                # get_observed_image_decomp(\
                #     N1=N1,N2=N2,basis=basis,solver=solver,\
                #     image=None, coeff_0=None, noise_scale=0,\
                #     Num_of_shapelets=Num_of_shapelets,alpha = alpha, \
                #     select_img_idx = select_img_idx, plot_decomp = True,\
                #     n_max = n_max)
                
                prep_and_do_noise_iter(image_data,\
                    noise_img, noise_img_num,noise_scale,\
                    N1=N1,N2=N2,basis=basis,solver=solver,\
                    image=None, coeff_0=None,\
                    Num_of_shapelets=Num_of_shapelets, alpha = alpha, \
                    select_img_idx = select_img_idx, \
                    plot_decomp = True,\
                    n_max = n_max)

                            
    elif(solver == 'omp'):
        
        ## Select number of shapelets that would be selected by OMP
        for Num_of_shapelets in Num_of_shapelets_array:
            
            ## Control the number of shapelets in initial decomposition for
            ## basis matrix - n_max controls the size of basis matrix
            
            if n_max < Num_of_shapelets:
                ## Just set this variable to something larger than
                ## Num_of_shapelets to control the number of shapelet vectors
                ## included in the basis
                n_max = 2*Num_of_shapelets
            for select_img_idx in selected_images:

                print("current image:\n")
                print(select_img_idx)
                
                # print("Getting observed decomposition")

                # obs_time0 = time.time()
                # get_observed_image_decomp(\
                #         N1=N1,N2=N2,basis=basis,solver=solver,\
                #         image=None, coeff_0=None, noise_scale=0,\
                #         Num_of_shapelets=Num_of_shapelets, \
                #         select_img_idx = select_img_idx, plot_decomp = True,\
                #         n_max = n_max)
                # print("Time taken observed image decomp: ")
                # print("%s seconds" % (time.time() - obs_time0))

                print("Doing noise iteration")

                noise_time0 = time.time()
                prep_and_do_noise_iter(image_data,\
                        noise_img, noise_img_num,noise_scale,\
                        N1=N1,N2=N2,basis=basis,solver=solver,\
                        image=None, coeff_0=None,\
                        Num_of_shapelets=Num_of_shapelets, \
                        select_img_idx = select_img_idx, \
                        plot_decomp = True,\
                        n_max = n_max)
                
                print("Time taken for noise iteration: ")
                print("%s seconds" % (time.time() - noise_time0))

    elif(solver == 'lstsq'):
        
        # get_observed_image_decomp(\
        #             N1=N1,N2=N2,basis=basis,solver=solver,\
        #             image=None, coeff_0=None, noise_scale=0,\
        #             Num_of_shapelets=Num_of_shapelets, \
        #             select_img_idx = select_img_idx, plot_decomp = True,\
        #             n_max = n_max)
        
        prep_and_do_noise_iter(image_data,\
                    noise_img, noise_img_num,noise_scale,\
                    N1=N1,N2=N2,basis=basis,solver=solver,\
                    image=None, coeff_0=None,\
                    Num_of_shapelets=Num_of_shapelets, \
                    select_img_idx = select_img_idx, 
                    plot_decomp = True,\
                    n_max = n_max)
        
    elif(solver == 'svd'):
        
        for Num_of_shapelets in Num_of_shapelets_array:
            
            for select_img_idx in selected_images:
                # get_observed_image_decomp(\
                #         N1=N1,N2=N2,basis=basis,solver=solver,\
                #         image=None, coeff_0=None, noise_scale=0,\
                #         Num_of_shapelets=Num_of_shapelets, \
                #         select_img_idx = select_img_idx, plot_decomp = True,\
                #         n_max = n_max)

                prep_and_do_noise_iter(image_data,\
                        noise_img, noise_img_num,noise_scale,\
                        N1=N1,N2=N2,basis=basis,solver=solver,\
                        image=None, coeff_0=None,\
                        Num_of_shapelets=Num_of_shapelets, \
                        select_img_idx = select_img_idx, \
                        plot_decomp = True,\
                        n_max = n_max)


if __name__=='__main__':
    
    Num_of_shapelets_array = [28]
    methods = ['lasso', 'omp', 'svd', 'lstsq']
    
    ## Range chose so that SNR is in range ~20 -- ~50
    min_noise = 5e-5; max_noise = 2e-4; noise_realizations = 5;
    step = (max_noise - min_noise) / noise_realizations

    noise_array = np.arange(min_noise, max_noise, step)
    alpha_ = np.logspace(-5,-1.3,6)
    basis_array = ['XY_Elliptical', 'Polar_Elliptical','Polar', 'XY', 'Compound_XY','Compound_Polar']

    # Generate noisy images
    # galsim images are 78 x 78
    size_X = 78; size_Y=78
    noisy_mat_num = 10

    noise_matrix = np.zeros((size_X*size_Y , noisy_mat_num))
    for i in range(noisy_mat_num):
        noise_matrix[:,i] = np.random.randn(size_X*size_Y)

    for noise_scale in noise_array:
        # Select a method for fitting the coefficients
        for basis in [basis_array[-1]]:
            
            for solver in ['omp']:#range(len(methods)): 

                test_stability(solver, basis, \
                        noise_scale, noise_img = noise_matrix, noise_img_num = noisy_mat_num,\
                        size_X = size_X, size_Y = size_Y,\
                        alpha_ = alpha_, Num_of_shapelets_array = Num_of_shapelets_array)
