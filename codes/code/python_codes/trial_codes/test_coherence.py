import numpy as np
import numpy.linalg as LA

# Norm of a vector with the Product func. as inner product
def Norm(A):
    S = reduce((lambda x, y: x+y**2), A,0)
    return np.sqrt(S)

# Normalize column vectors
def Normalize(A_):
    A = A_
    Norm_A = LA.norm(A) 
    A = map((lambda x: x/Norm_A),A)
    return A

# Get a matrix and calculate coherence
def Coherence(A):

    B = np.zeros_like(A)

    for i in range(np.shape(A)[1]):
        B[:, i] = Normalize(A[:,i])
    
    Product = np.dot(B.transpose(), B)
    return (Product - np.identity(np.shape(Product)[0])).max() 
# Doesn't mater for the shape which one I take, Product is a square matrix
