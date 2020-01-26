import galsim
import numpy as np
from multiprocessing import Pool

def generateGaussianTrainingSet():
    img_size = 32
    destination = 'testImages/'
    #sizes = np.array([3.,3.5,4.,4.5,5.,5.5,6.])
    sizes = np.array([2.5,3.75,4.25,5.35,5.9,6.2])
    N_sizes = len(sizes)
    N_shears = 4 #16
    N_angles = 10#72
    N_train = N_sizes*N_shears*N_angles
    if N_train < img_size**2:
        print("Too few training images to generate an \
               overcomplete dictionary")

    pos_angles = np.linspace(0.,180.,N_angles)
    shears = np.arange(0.,1.,1./N_shears)

    print("No. of training images = ", N_train)

    img = galsim.Image(img_size,img_size)
    print("Generating training images ... ")

    gal_flux = 1.e5
    noise = 30

    for N_size in xrange(N_sizes):
      for N_shear in xrange(N_shears):
        for N_angle in xrange(N_angles):
                g = galsim.Gaussian(sigma=sizes[N_size],flux=gal_flux)
                theta = galsim.Angle(pos_angles[N_angle],galsim.degrees)
                g = g.shear(g=shears[N_shear],beta=theta)
                g.drawImage(image=img)
                img.addNoise(galsim.GaussianNoise(sigma=noise))
                fname = 'test_'+str(N_size).zfill(2)+'_'+str(N_shear).zfill(2)+'_'+str(N_angle).zfill(2)
                img.write(fname+'.fits',dir=destination)

    print("Finished generating "+str(N_train)+" training images.")
    print("Destination folder: "+str(destination))

def generateBDTrainingSet(btt):
    gsp = galsim.GSParams(maximum_fft_size=16384)
    img_size = 64
    destination = 'trainImages/'
    
    # PSF description
    lam = 1580 #nm
    diam = 2.4 #m
    obscuration = 0.3 
    PSF = galsim.OpticalPSF(lam=lam,diameter=diam,obscuration=obscuration)

    # Bulge component
    n_bulge = 4.0
    bulge_Re_arr = np.array([3.5,4.,4.5,5.,5.5,6.,6.5]) # pixels	
    #sizes = np.array([2.5,3.75,4.25,5.35,5.9,6.2])

    #Disk component
    n_disk = 1.0
    disk_scaleRadius_arr = bulge_Re_arr

    N_sizes = len(bulge_Re_arr)*len(disk_scaleRadius_arr)
    N_shears = 4 #16
    N_angles = 10#72

    N_train = N_sizes*(N_shears**2)*(N_angles**2)
    if N_train < img_size**2:
        print("Too few training images to generate an \
               overcomplete dictionary")

    theta_values = np.linspace(0.,180.,N_angles).tolist() # degrees
    pos_angles = [galsim.Angle(theta,galsim.degrees) for theta in theta_values]
    shears = np.arange(0.,1.,1./N_shears)

    print("No. of training images = ", N_train)

    img = galsim.Image(img_size,img_size)
    print("Generating training images ... ")

    gal_flux = 5.e5
    noise = 10

    for N_bulge_size in xrange(len(bulge_Re_arr)):
        for N_disk_size in xrange(len(disk_scaleRadius_arr)):
          bulge = galsim.Sersic(n=n_bulge,half_light_radius=bulge_Re_arr[N_bulge_size],gsparams=gsp)
          disk = galsim.Sersic(n=n_disk,scale_radius=disk_scaleRadius_arr[N_disk_size],gsparams=gsp)
          for N_bulge_shear in xrange(N_shears):
            for N_bulge_pos in xrange(N_angles):
              B = bulge.shear(g=shears[N_bulge_shear],beta=pos_angles[N_bulge_pos])
              for N_disk_shear in xrange(N_shears):
                for N_disk_pos in xrange(N_angles):
                  D = disk.shear(g=shears[N_disk_shear],beta=pos_angles[N_disk_pos])
                  gal = btt*gal_flux*B+(1.-btt)*gal_flux*D
                  gal = galsim.Convolve([gal,PSF])
                  gal.drawImage(image=img)
                  img.addNoise(galsim.GaussianNoise(sigma=noise))
                  fname = 'trainBD_'+str(N_btt).zfill(2)+'_'+str(N_bulge_size).zfill(2)+'_'+str(N_disk_size).zfill(2)+\
                          '_'+str(N_bulge_shear).zfill(2)+'_'+str(N_bulge_pos).zfill(2)+'_'+str(N_disk_shear).zfill(2)+\
                          '_'+str(N_disk_pos).zfill(2)
                  img.write(fname+'.fits',dir=destination)

    print("Finished generating "+str(N_train)+" training images with bulge+disk components")
                                                     
def generateTestingSet():
    img_size = 64
    N_sizes = 9
    N_angles = 36
    sizes = np.linspace(4.,8.,N_sizes)

if __name__=='__main__':
    BTT = np.arange(0.05,1.,0.05) # 0.05, 0.1, ..., 0.95
    pool1 = Pool(len(BTT))
    pool1.map(generateBDTrainingSet,BTT.tolist())
