------------
Folder names
@*parent_folder*
------------

Folder names refer to the simulated *noise_scale* as 2nd part in @decompositions

-------------
Stability plots
@stability
-------------

Names are constructed as follows:
    
    1st part:
        -- solver - as in 1st part of @decompositions 
    2nd part:
        -- For lasso method - alpha coeff as in 6th part @decomp
        -- For sparse method - Number of shapelets allowed for decomp as in 6th part @decomp
        -- For lstsq method - n_max as in 4th part @decomp
    3rd part:
        -- basis - selected basis of decomposition
    4th part:
        -- *signal_to_noise* - refers to the signal to noise of the selected image

-------------
Thise refers to the decomposition plots
@decompositions
-------------
Image names are formated as following:  

1st part of the string is the solver included:  
    -- *lstsq_* - least squares solution  
    -- *lasso_* - lasso regularization method  
    -- *SVD_* - singular value decomposition  
    -- *Sparse_* - Orthogonal matching pursuit method (OMP)  

2nd part refers to noise scale of the imag. That is, constant in front of the random matrix:  
    -- *_noise_scale_*  
  
3rd part of the string says what are N1 and N2 respectively. These control the maximum order of  
shapelets included into the decomposition. For example:  
    -- *_20_20_* - this means N1 = 20 and N2 = 20  

4th part is about upper limit to the shapelet order that can be included:  
    -- *_n_max_*  
    
5th part is describing the coordinate system in which the decomposition was made. For example:  
    -- *_Polar_* - polar coordinate space for shapelets  
    -- *_XY_* - descartes coordinate space for shapelets  
    -- *_Elliptic_* - elipse coordinate space for shapelets (still not implemented)  

-------------

6th part:  
    For OMP:  
    -- *_number_of_shapelets_to_be_used_* - it describes the maximum number of shapelets used in OMP  
    For lasso:  
    -- *_alpha_* - coefficient in front of l_1 norm of the coefficients  
