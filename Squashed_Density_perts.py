import matplotlib
matplotlib.use('agg')
import numpy as np
import numpy.linalg as LA
import time
import numpy.fft
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

pi = np.pi

###################
# input parameters, read from command line
###################
parser = OptionParser()
parser.add_option('--kmin', dest='kmin', 
                  help='minimum wavenumber.', 
                  default=2)
parser.add_option('--kmax', dest='kmax', 
                  help='maximum wavenumber.', 
                  default=20)
parser.add_option('--size', dest='size', 
                  help='size of each direction of data cube.  default=256', 
                  default=256)
parser.add_option('--alpha', dest='alpha', 
                  help='negative of power law slope.  (Power ~ k^-alpha) '+
                  'supersonic turbulence is near alpha=2.  '+
                  'driving over a narrow band of two modes '+
                  'is often done with alpha=0', 
                  default = 1)
parser.add_option('--seed', dest='seed', 
                  help='seed for random # generation.  default=0', 
                  default = 0)
parser.add_option('--ampl', dest='ampl', 
                  help='Amplitude of the density perturbation.  default=1', 
                  default = 1)

(options, args) = parser.parse_args()

# size of the data domain
n = [int(options.size)+8, int(options.size)+8, int(options.size)+8] # the +8s are ghost zones to match athena
# range of perturbation length scale in units of the smallest side of the domain
kmin = int(options.kmin)
kmax = int(options.kmax)
if kmin > kmax or kmin < 0 or kmax < 0:
    print "kmin must be < kmax, with kmin > 0, kmax > 0.  See --help."
    sys.exit(0)
if kmax > np.floor(np.min(n))/2:
    print "kmax must be <= floor(size/2).  See --help."
    sys.exit(0)
alpha = options.alpha
if alpha==None:
    print "You must choose a power law slope, alpha.  See --help."
    sys.exit(0)
alpha = float(options.alpha)
if alpha < 0.:
    print ("alpha is less than zero. That means there is more power on small "
           "scales!.  See --help.")
    sys.exit(0)
seed = int(options.seed)
ampl = float(options.ampl)
# data precision
dtype = np.float64

np.random.seed(seed=seed)

###################
# begin computation
###################


kx = np.zeros(n, dtype=dtype)
ky = np.zeros(n, dtype=dtype)
kz = np.zeros(n, dtype=dtype)
# perform fft k-ordering convention shifts
for j in range(0,n[1]):
    for k in range(0,n[2]):
        kx[:,j,k] = n[0]*np.fft.fftfreq(n[0])
for i in range(0,n[0]):
    for k in range(0,n[2]):
        ky[i,:,k] = n[1]*np.fft.fftfreq(n[1])
for i in range(0,n[0]):
    for j in range(0,n[1]):
        kz[i,j,:] = n[2]*np.fft.fftfreq(n[2])

        
kx = np.array(kx, dtype=dtype)
ky = np.array(ky, dtype=dtype)
kz = np.array(kz, dtype=dtype)
k = np.sqrt(np.array(kx**2+ky**2+kz**2, dtype=dtype))
# only use the positive frequencies
inds = np.where(np.logical_and(k**2 >= kmin**2, k**2 < (kmax+1)**2))
nr = len(inds[0])
phasex = np.zeros(n, dtype=dtype)
phasex[inds] = 2.*pi*np.random.uniform(size=nr)
fx = np.zeros(n, dtype=dtype)
fx[inds] = np.random.normal(size=nr)


for i in range(kmin, kmax+1):
    slice_inds = np.where(np.logical_and(k >= i, k < i+1))
    rescale = np.sqrt(np.sum(np.abs(fx[slice_inds])**2 ))/n[0]
    fx[slice_inds] = fx[slice_inds]/rescale

fx[inds] = fx[inds]*k[inds]**-(0.5*alpha)
fx = np.cos(phasex)*fx + 1j*np.sin(phasex)*fx

pertx = np.real(np.fft.ifftn(fx))
pertx = pertx-np.average(pertx)
STD = np.std(pertx)
Pow10 = np.ceil(np.log10(STD))
Normalization = (np.around(STD, -int(Pow10-2.))/ampl)
pertx /= Normalization


for kk in xrange(pertx.shape[0]):
    for j in xrange(pertx.shape[0]):
      for i in xrange(pertx.shape[0]):
          if pertx[i,j,kk] >= 0.99:
              pertx[i,j,kk] = 0.99
          if pertx[i,j,kk] <= -0.99:
              pertx[i,j,kk] = -0.99
    









def angular_distance(t1,p1, t2,p2):
    return np.arccos(np.cos(t1)*np.cos(t2)+np.sin(t1)*np.sin(t2)*np.cos(np.absolute(p1-p2)))

def spherical_distance(r1,t1,p1, r2,t2,p2):
    return np.sqrt(r1**2 + r2**2 - 2*r1*r2*(np.cos(t1)*np.cos(t2)+np.sin(t1)*np.sin(t2)*np.cos(np.absolute(p1-p2))))

def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

rmax = 3.15
rout = 2.0
c = 23./6. # compression factor, so everything between 2 and 3.15 will now be compressed down to 2 + (3.15-2)/c
rmaxnew = rout + (rmax-rout)/c

NRES = n[0]

U = np.zeros((4,NRES,NRES,NRES))
r = np.zeros((NRES,NRES,NRES))
Target  = np.zeros((5,NRES,NRES,NRES)) # 0 = f, 1 = theta, 2 = phi, 3 = sum of dperts, 4 = num of dperts 
TBC     = np.zeros((4,NRES,NRES,NRES)) # 0 = f, 1 = theta, 2 = phi, 3 = dpert

for k in xrange(NRES):
    for j in xrange(NRES):
        for i in xrange(NRES):
            U[1,k,j,i] = rmax*(i-(NRES/2. - 0.5)) / (NRES/2. - 0.5)
            U[2,k,j,i] = rmax*(j-(NRES/2. - 0.5)) / (NRES/2. - 0.5)
            U[3,k,j,i] = rmax*(k-(NRES/2. - 0.5)) / (NRES/2. - 0.5)
            U[0,k,j,i] = pertx[k,j,i]
            r[k,j,i] = LA.norm([U[1,k,j,i], U[2,k,j,i],U[3,k,j,i]])
            if r[k,j,i] >= rout:
                TBC[0,k,j,i] = (r[k,j,i]-rout)/(rmax-rout)
                TBC[1,k,j,i] = np.arccos(U[3,k,j,i]/r[k,j,i])
                TBC[2,k,j,i] = np.arctan2(U[2,k,j,i],U[1,k,j,i])
                TBC[3,k,j,i] = U[0,k,j,i]
                if r[k,j,i] <= rmaxnew :
                    Target[0,k,j,i] = (r[k,j,i]-rout)/(rmaxnew-rout)
                    Target[1,k,j,i] = np.arccos(U[3,k,j,i]/r[k,j,i])
                    Target[2,k,j,i] = np.arctan2(U[2,k,j,i],U[1,k,j,i])


"""
for each cell in the target region calculate the volume in the TBC that it corresponds to. 
c is the contraction factor
if all cells are size d x d x d then i will approximate it by a puck of thickness d and radius d
if my target cell is at r, s.t. f = (r-rout)/(rmaxnew-rout), so the puck is r-d/2, r+d/2 in the radial direction
then the region that corresponds to it in the TBC is centered in the radial direction at r = f*(rmax-rout)+rout
the minimum r of the region is r = f_-*(rmax-rout)+rout where f_- = (r-d/2-rout)/(rmaxnew-rout)
and the max r of the region is r = f_+*(rmax-rout)+rout where f_+ = (r+d/2-rout)/(rmaxnew-rout)
or in short the new region is bound between f*(rmax-rout)+rout - c*d/2 and f*(rmax-rout)+rout + c*d/2

now the angular extent.
the half opening angle should be theta = np.arctan2(d,r)
So this defines my region

My region can be defined by looking for all points in:
    i)   the correct subregion
    ii)  the correct radius bins
    iii) are closer to the r_target vector than d*(r_tbc / r_target)


"""
time_start = time.time()

def d_to_line(x,p):
    return( LA.norm(np.cross(p,p-x))/LA.norm(x))

NZT = np.nonzero(Target[1,...])

Target_non_zero_indices    = np.array([ [NZT[2][i]   
                                        ,NZT[1][i]   
                                        ,NZT[0][i]]    
                                        for i in xrange(len(NZT[0]))])
N_target = len(Target_non_zero_indices)
d = np.mean(np.diff(U[1,...]))


time_start = time.time()




# for l in xrange(N_target):
l = rank
while l < N_target:
    f = (r[Target_non_zero_indices[l][2],Target_non_zero_indices[l][1],Target_non_zero_indices[l][0]]-rout)/(rmaxnew-rout)
    k  = Target_non_zero_indices[l][2]
    j  = Target_non_zero_indices[l][1]
    i  = Target_non_zero_indices[l][0]
    Ks = [k]
    Js = [j]
    Is = [i]
    for jj in xrange(-1,2):
        for ii in xrange(-1,2):
            if j+jj == NRES/2:
                continue
            else:
                Is.append(min(np.abs(i+ii-NRES/2.)/np.abs(j+jj-NRES/2.), 1) * NRES/2* np.sign(i+ii-NRES/2.) +NRES/2.)
    for kk in xrange(-1,2):
        for ii in xrange(-1,2):
            if k+kk == NRES/2:
                continue
            else:
                Is.append(min(np.abs(i+ii-NRES/2.)/np.abs(k+kk-NRES/2.), 1) * NRES/2* np.sign(i+ii-NRES/2.) +NRES/2.)
    for ii in xrange(-1,2):
        for jj in xrange(-1,2):
            if i+ii == NRES/2:
                continue
            else:
                Js.append( min(np.abs(j+jj-NRES/2.)/np.abs(i+ii-NRES/2.), 1) * NRES/2* np.sign(j+jj-NRES/2.) +NRES/2.)
    for kk in xrange(-1,2):
        for jj in xrange(-1,2):
            if k+kk == NRES/2:
                continue
            else:
                Js.append( min(np.abs(j+jj-NRES/2.)/np.abs(k+kk-NRES/2.), 1) * NRES/2* np.sign(j+jj-NRES/2.) +NRES/2.)
    for jj in xrange(-1,2):
        for kk in xrange(-1,2):
            if j+jj == NRES/2:
                continue
            else:
                Ks.append(min(np.abs(k+kk-NRES/2.)/np.abs(j+jj-NRES/2.), 1) * NRES/2* np.sign(k+kk-NRES/2.) +NRES/2.)
    for ii in xrange(-1,2):
        for kk in xrange(-1,2):
            if i+ii == NRES/2:
                continue
            else:
                Ks.append(min(np.abs(k+kk-NRES/2.)/np.abs(i+ii-NRES/2.), 1) * NRES/2* np.sign(k+kk-NRES/2.) +NRES/2.)
    imin=max(int(np.ceil(np.min(Is)))-1,0)
    imax=min(int(np.ceil(np.max(Is)))+1,NRES)
    jmin=max(int(np.ceil(np.min(Js)))-1,0)
    jmax=min(int(np.ceil(np.max(Js)))+1,NRES)
    kmin=max(int(np.ceil(np.min(Ks)))-1,0)
    kmax=min(int(np.ceil(np.max(Ks)))+1,NRES)
    for kk in xrange(kmin,kmax):
        for jj in xrange(jmin,jmax):
            for ii in xrange(imin,imax):
                if r[kk,jj,ii] >= f*(rmax-rout)+rout - c*d/2. and r[kk,jj,ii] <= f*(rmax-rout)+rout + c*d/2.:
                    if ( d_to_line(U[1:4,k,j,i], U[1:4,kk,jj,ii]) <= d * (r[kk,jj,ii]/r[k,j,i])):
                        Target[3,k,j,i] += TBC[3,kk,jj,ii]
                        Target[4,k,j,i] += 1
    l += size




comm.Barrier()
if comm.rank==0:
    # only processor 0 will actually get the data
    totals3 = np.zeros_like(Target[3,...])
    totals4 = np.zeros_like(Target[4,...])
else:
    totals3 = None
    totals4 = None

# use MPI to get the totals 
comm.Reduce(
    [Target[3,...], MPI.DOUBLE],
    [totals3, MPI.DOUBLE],
    op = MPI.SUM,
    root = 0
)

# use MPI to get the totals 
comm.Reduce(
    [Target[4,...], MPI.DOUBLE],
    [totals4, MPI.DOUBLE],
    op = MPI.SUM,
    root = 0
)



if comm.rank==0:
    UNEW = np.zeros((4,NRES,NRES,NRES))
    for k in xrange(NRES):
        for j in xrange(NRES):
            for i in xrange(NRES):
                if r[k,j,i] > rout and r[k,j,i] <= rmaxnew and totals4[k,j,i] != 0.0:
                    # print totals3[k,j,i], totals4[k,j,i]
                    UNEW[0,k,j,i] = totals3[k,j,i]/totals4[k,j,i]
                elif r[k,j,i] < rout:
                    UNEW[0,k,j,i] = U[0,k,j,i]
                else:
                    UNEW[0,k,j,i] =0.
    fig, axes = plt.subplots(nrows=1, ncols=2)
    im1 = axes[0].pcolor(U[1,n[0]/2,:,:],U[2,n[0]/2,:,:],(UNEW[0,n[0]/2,:,:]), vmin = np.min(U[0,n[0]/2,:,:]),vmax = np.max(U[0,n[0]/2,:,:]), cmap='Spectral')
    axes[0].plot(d/2+np.linspace(-2,2),d/2+np.sqrt(2**2 - np.linspace(-2,2)**2), 'k' )
    axes[0].plot(d/2+np.linspace(-2,2),d/2+-np.sqrt(2**2 - np.linspace(-2,2)**2), 'k' )
    im2 = axes[1].pcolor(U[1,n[0]/2,:,:],U[2,n[0]/2,:,:],(U[0,n[0]/2,:,:]), vmin = np.min(U[0,n[0]/2,:,:]),vmax = np.max(U[0,n[0]/2,:,:]),cmap='Spectral')
    axes[1].plot(d/2+np.linspace(-2,2),d/2+np.sqrt(2**2 - np.linspace(-2,2)**2), 'k' )
    axes[1].plot(d/2+np.linspace(-2,2),d/2+-np.sqrt(2**2 - np.linspace(-2,2)**2), 'k' )
    # plt.show()
    plt.savefig('squashed'+str(n[0]-8)+'_seed_'+str(seed)+'_ampl_'+str(ampl)+'.png')
    ########### save the pert file ##########################################
    f = open('dpert_n_'+str(n[0]-8)+'_seed_'+str(seed)+'_ampl_'+str(ampl)+'.dat','w')
    f.write('kmin = %i kmax = %i size = %i \n'  %(kmin, kmax, n[0]-8))
    f.write('alpha = %f seed = %i ampl = %f \n \n'  %(alpha, seed, ampl))
    f.write('n = %d \n \n '  %(np.product(n)))
    for k in xrange(n[0]):
        for j in xrange(n[0]):
            for i in xrange(n[0]):
                f.write('\t %d \t %d \t %d \t %f \n' %(k,j,i, UNEW[0,k,j,i] ))
    f.close()
    #########################################################################

