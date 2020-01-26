import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import gaussian_kde
from utils.I_O_utils import *

#import pdb; pdb.set_trace()

def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    
    @param mypath - local path 
    
    --------------
    Change the root_path in accordance to your wished root path
    """

    from errno import EEXIST
    from os import makedirs,path
    import os

    root_path = '/home/kostic/Documents/codes/code/python_codes/'
    
    try:
        makedirs(root_path + mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(root_path + mypath):
            pass
        else: raise

def _visualize(\
        coeffs_val_cluster, label_arr_cluster,\
        basis,solver,\
        beta_array,\
        N1=20,N2=20):

    """
    Bin the coefficient values and plot the result. Account for only nonzero values in the binning.

    Parameters:
    -----------
    coeffs_val_cluster : values of the coefficients to be binned
    label_arr_cluster : labels of the shapelets for the binning
    basis : basis in which the decomposition was done
    solver : solver used for fitting the coefficients
    beta_array : array of beta values used for different basis
    
    Optional:
    ---------
    N1,N2 : dimension of the shapelet basis is N1*N2
    """

    ## Number of galaxies included in the statistics
    galaxy_num = coeffs_val_cluster.shape[0]

    ## Every coeffs array is going to have same label indexation
    f_path = 'Plots/Cluster_test/Beta_Scales/' + solver + '/'

    step = 0
    for beta in beta_array:
        
        str_beta = str("%.3f" % (beta))
        
        mkdir_p(f_path + str_beta +'/')
        
        left = step*(N1*N2)
        right = (step+1)*(N1*N2)
        coeffs_val_curr = coeffs_val_cluster[:, left:right]

        indices = np.where(coeffs_val_curr!=0); step+=1;
    
        ## Get the integer numbers representing shapelet labels
        ## for corresponding coefficient values
        idx_1 = indices[0]
        idx_2 = indices[1] 

        ## Get number of distinct shapelets
        ## along with the basis index, to know from which basis
        ## they come from
        sh_labels = []
        basis_labels = []
        for i in xrange(len(idx_2)):
            sh_label = idx_2[i]
            if not(sh_label in sh_labels):
                sh_labels.append(sh_label)
                basis_labels.append(idx_1[i])

        sh_labels = np.asarray(sh_labels)
     
        print("idx_1\n", idx_1)
        print("idx_2\n", idx_2)
        print("sh_labels\n", sh_labels)
        print("basis_labels\n", basis_labels)
        
        for i in xrange(len(sh_labels)):
            sh_label = sh_labels[i]
            basis_label = basis_labels[i]

            fig, ax = plt.subplots()

            ax.set_title('Shapelet ' + label_arr_cluster[basis_label][sh_label])
            
            data_hist = coeffs_val_curr[:, sh_label]
            
            ## Count into the bins only nonzero values
            weights = np.zeros_like(data_hist)
            weights[np.where(data_hist != 0)] = 1

            H, bins, patches = ax.hist(data_hist, \
                    weights = weights, histtype = 'step', align = 'mid')  
            
            ## Set up y labels
            ax.set_yticks(np.linspace(0., galaxy_num, num = 11))
            y_tick_labels = [val/galaxy_num for val in ax.get_yticks()]
            ax.set_ylabel('Fraction of galaxies')
            ax.set_yticklabels(y_tick_labels)

            ## Set up x labels
            few_biggest = 2
            idx_few_biggest = H.argsort()[-few_biggest:]
            for idx in idx_few_biggest:
                height = H[idx]
                if height != 0:
                    x_coord = 0.5*(bins[idx] + bins[idx+1])
                    y_coord = height + 0.05*galaxy_num
                    ax.text(x_coord, y_coord, str("%.2e" % (x_coord)), \
                        fontsize = 'small',ha = 'center', va='bottom')
            
            ax.ticklabel_format(stype='sci', axis = 'x', scilimits = (0,0)) 
            ax.set_xlabel('Coefficient values')
            
            file_save_path = f_path + str_beta+'/' \
                + label_arr_cluster[basis_label][sh_label]
            
            ## Save file with all the bin values
            ## just for check
            f = open(file_save_path + '.txt','w')
            f.write("Bin value\tBin center value\n")
            nonzero_bins = np.where(H != 0)[0]
            for i in nonzero_bins:
                f.write("%.1f\t%.5f\n" % (float(H[i])/galaxy_num, 0.5*(bins[i] + bins[i+1])))
            f.close()

            plt.tight_layout()
            plt.savefig(file_save_path + '.png')
            plt.cla()
            plt.clf()
            plt.close()

def _dist_matrix_angle_mat(coeffs_matrix,beta_array, mid_word = '', \
        cmap_dist = mpl.cm.gray_r, cmap_dot = mpl.cm.bwr):

    k = 0
    for beta in beta_array:

        beta_str = str("%.3f" % (beta))
        galaxy_vectors = np.zeros((100,55))
        galaxy_vectors = coeffs_matrix[:, k, :]; 
        
        print("beta %f:\n" % (beta))
        print("Similarity between galaxy_vectors and coeffs_matrix:\n")
        print(np.allclose(galaxy_vectors, coeffs_matrix[:,k,:]))
        
        k+=1
        
        dist_normed_matrix = np.zeros((100,100))
        dot_matrix = np.zeros((100,100))

        for i in xrange(100):
            for j in xrange(100):
                norm1_2 = np.dot(galaxy_vectors[i], galaxy_vectors[i])
                norm2_2 = np.dot(galaxy_vectors[j], galaxy_vectors[j])

                if (norm1_2 !=0) and (norm2_2 !=0):
                    if norm1_2 > norm2_2:
                        norm = norm2_2
                    else:
                        norm = norm1_2
                    diff = (galaxy_vectors[i] - galaxy_vectors[j])
                    dist_normed_matrix[i,j] = np.sqrt(np.dot(diff, diff)) / np.sqrt(norm) 
                    dot_matrix[i,j] = np.dot(galaxy_vectors[i], galaxy_vectors[j]) / (np.sqrt(norm1_2) * np.sqrt(norm2_2))
                else:
                    dot_matrix[i,j] = -1.5
                    dist_normed_matrix[i,j] = -1.

        norm_dist = mpl.colors.Normalize(\
                #linthresh = 1.,
                vmin = np.min(dist_normed_matrix),
                vmax = np.max(dist_normed_matrix))
        
        norm_dot = mpl.colors.Normalize(\
                vmin = np.min(dot_matrix),
                vmax = np.max(dot_matrix))

        s_m_dist = cm.ScalarMappable(cmap=cmap_dist, norm=norm_dist); s_m_dist.set_array([])
        s_m_dot = cm.ScalarMappable(cmap=cmap_dot, norm=norm_dot); s_m_dot.set_array([])

        fig, ax = plt.subplots()
         
        ax.set_xlabel('galaxy_idx')
        ax.set_ylabel('galaxy_idx')
        
        ## Force colorbars besides the axes
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size='5%')    

        im = ax.imshow(dist_normed_matrix, cmap=cmap_dist ,interpolation ='none')  
        fig.colorbar(s_m_dist,cax=cax,format = '%.2e', orientation='vertical')
        plt.savefig('testing/test_distance_normed_'\
                +mid_word +'_'\
                +beta_str + '_.png')
        plt.clf()

        fig, ax = plt.subplots()
         
        ax.set_xlabel('galaxy_idx')
        ax.set_ylabel('galaxy_idx')
        
        ## Force colorbars besides the axes
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size='5%')    


        im = ax.imshow(dot_matrix, cmap=cmap_dot, interpolation='none')
        fig.colorbar(s_m_dot, cax=cax, format='%.2e', orientation='vertical')
        plt.savefig('testing/test_dot_'\
                +mid_word+'_'\
                +beta_str+'_.png')
        plt.close('all')


def _SOM(data, labels,\
        beta_array,\
        mid_word = '',\
        n_job = 1,\
        verbose = 'info',\
        mapsize = [20,20],\
        train_rough_len = 500,train_finetune_len=100,\
        normalization='var', initialization='pca',cmap = mpl.cm.bwr):
    
    import sompy_custom as sompy_c
    from sompy_custom.sompy import SOMFactory
    from sompy_custom.visualization.mapview import View2D
        
    som = SOMFactory.build(data, \
        mapsize=mapsize, normalization=normalization, initialization=initialization,\
        component_names=labels)
    som.train(n_job=n_job, \
        verbose=verbose, \
        train_rough_len=train_rough_len, train_finetune_len=train_finetune_len)
    
    print("Topographic error")
    print(som.calculate_topographic_error())

    print("Final quantization error")
    print(np.mean(som._bmu[1]))

    print("\n")
    print("View_2D in process ...\n")

    map_size = som.calculate_map_size('rect')
    print("Map size\n")
    print(map_size, "\n")
    
    str_mapsize_2 = str("%d" % (mapsize[1]))
    str_mapsize_1 = str("%d" % (mapsize[0]))

    view2D = View2D(10, 10, mid_word,text_size=10)
    view2D.show(som,col_sz=4, cmap=cmap, which_dim='all',desnormalize=True)
    view2D.save('testing/SOM_' + mid_word + '_' + str_beta + '_' + str_mapsize_1 + '_' + str_mapsize_2 +'_.png')
    plt.clf()

    print("UMatrix in process ...\n")

    u_mat = sompy_c.umatrix.UMatrixView(10, 10, 'umatrix',\
            show_axis=True, text_size=8, show_text=True)
    u_MAT = u_mat.build_u_matrix(som, distance=1, row_normalized=False)
    u_MAT = u_mat.show(som, \
            distance2=1, row_normalized=False, show_data=True, contooor=True,blob=False)
    plt.savefig('testing/SOM_U_mat_' + mid_word + '_' + str_beta +'_' + str_mapsize_1 + '_' + str_mapsize_2 +'_.png')
    plt.close('all')

def _MDS(galaxy_vectors, \
        n_components=2, max_iter=3000, metric=True, verbose=1,\
        dissimilarity='euclidean',\
        mid_word=''):
    
    from sklearn import manifold

    print("####################")
    print("####### MDS ########")
    print("####################")
    
    dist_matrix = np.zeros((100,100))

    for i in xrange(100):
        for j in xrange(100):
            norm1_2 = np.dot(galaxy_vectors[i], galaxy_vectors[i])
            norm2_2 = np.dot(galaxy_vectors[j], galaxy_vectors[j])
            diff = (galaxy_vectors[i] - galaxy_vectors[j])
            dist_matrix[i,j] = np.sqrt(np.dot(diff, diff)) 
                
    print(dist_matrix)

    mds = manifold.MDS(\
            n_components=n_components, metric=metric, \
            max_iter=max_iter,verbose=verbose,\
            dissimilarity = dissimilarity)
    Y = mds.fit_transform(dist_matrix)
    plt.figure(figsize=(25,25)) 
    plt.scatter(Y[:,0], Y[:,1])
    idx=0
    for x,y in zip(Y[:,0], Y[:,1]):
        plt.text(x,y, str(idx), color='red', fontsize=12)
        idx+=1
    plt.xlabel('Dimension_1')
    plt.ylabel('Dimension_2')
    plt.savefig('testing/MDS_' + mid_word + '_.png')
    plt.clf()
    plt.close('all')


if __name__ == '__main__':

    coeffs_val_cluster = []; label_arr_cluster=[]
    basis = 'Compound_Polar'; solver='omp'
    Num_of_shapelets = 28; str_Num_of_shapelets = str("%d" % (Num_of_shapelets))
    
    _dist_flag = 0
    _som_flag = 1
    _mds_flag = 0
    _cluster_flag = 0
    
    f = open('data_cluster_'+basis+'_'+ str_Num_of_shapelets + '_.txt', 'r')
    for line in f:
        data = line.split()
        if len(data) == 3:
            if len(data[0]) == 3:
                if (data[0][0] == '(') and (ord(data[0][1]) <= 57 and ord(data[0][1]) >= 48):
                    coeffs_val_cluster.append(float(data[2]))
                    label_arr_cluster.append(data[0] + data[1])
    f.close()
    coeffs_val_cluster = np.asarray(coeffs_val_cluster)
    label_arr_cluster = np.asarray(label_arr_cluster)
    
    print(coeffs_val_cluster.shape)
    
    coeffs_matrix = coeffs_val_cluster.reshape((100,5,55))
    label_matrix = label_arr_cluster.reshape((100,5,55))
    beta_array = [1.881, 2.097, 2.531, 3.182, 4.918]
    
    if _dist_flag:
        _dist_matrix_angle_mat(coeffs_matrix, beta_array, \
                mid_word=basis+'_'+str_Num_of_shapelets) 
   
    if _mds_flag:
        galaxy_vectors = coeffs_val_cluster.reshape((100, 5*55))
        _MDS(galaxy_vectors, \
            n_components=2, max_iter=3000, metric=True, verbose=1,\
            mid_word=basis+'_'+str_Num_of_shapelets+'_' +solver)
    if _som_flag:
            galaxy_vectors = coeffs_val_cluster.reshape((100, 5*55))
            _SOM(\
                galaxy_vectors, label_matrix[0,0,:], beta_array,\
                mid_word = basis + '_' + str_Num_of_shapelets,
                n_job = 1,verbose = 'info', mapsize = [2,2],\
                train_rough_len = 500,train_finetune_len=500,\
                initialization='pca',cmap = mpl.cm.bwr)  
    if _cluster_flag:
        import fastcluster
        
    #_make_sp_matrix(coeffs_matrix)
