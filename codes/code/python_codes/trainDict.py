import glob
import numpy as np
from sklearn.decomposition import dict_learning as DL
import pyfits
import pickle

K = 80
alpha = 0.1
size = 32
fnames = glob.glob("trainImages/train_*")
nData = len(fnames)
print "nData = ", nData
Data = np.zeros((nData,size**2))
for i in xrange(nData):
    fn = fnames[i]
    img = pyfits.getdata(fn)
    Data[i,:] = img.ravel()
    if i%100==0:
        print i
        
D = DL(Data,K,alpha,max_iter=100,tol=1.e-2,verbose=1)
pickle.dump(D,open('sklearn_D.p','wb'))
