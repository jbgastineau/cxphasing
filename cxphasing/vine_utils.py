#!/usr/bin/env python
#
#       vine_utils.py
#       
#       Copyright 2009 David Vine <djvine@gmail.com>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

import sys
import scipy as sp
import numpy as np
import scipy.fftpack as spf
import scipy.constants as spc
import pdb
import tkFileDialog
import pylab as pyl
import scipy.ndimage.fourier as spnf
import readMDA

def get_path(filename):
    return '/'.join(filename.split(filename)[:-1])

def max(arr):
    maxi = np.argmax(arr)
    return sp.unravel_index(maxi,arr.shape), arr.max()
    

def fromfile(filename, dtype=sp.float64):
    tmp = sp.fromfile(filename, dtype)
    p= None
    if tmp.shape[0] == 128**2.:
        p=128
        tmp = tmp.reshape(128,128)
    elif tmp.shape[0] == 256**2.:
        p=256
        tmp = tmp.reshape(256, 256)
    elif tmp.shape[0] == 512**2.:
        p=512
        tmp = tmp.reshape(512, 512)
    elif tmp.shape[0] == 1024**2.:
        p=1024
        tmp = tmp.reshape(1024, 1024)
    elif tmp.shape[0] == 2048**2.:
        p=2048
        tmp = tmp.reshape(2048, 2048)
    if p != None:
        print 'Array shape auto-detect: %i' % p
    else:
        print 'Did not determine array shape'
    return tmp

def radial_average(arr1):
    pix = arr1.shape[0]
    # first compute the radius for each pixel in the image
    y, x = np.indices(arr1.shape) # y, x have same shape as im, and values in each
                            # are their respective y and x locations.
    pix_rad = np.sqrt((x-pix/2)**2 + (y-pix/2)**2) # radius for each pixel in im
    ind = np.argsort(pix_rad.flat) # indices for sorted radii (need to use with im)
    sorted_radii = pix_rad.flat[ind] # the sorted radii
    sorted_im = arr1.flat[ind] # image values sorted by radii
    pix_rad_int = sorted_radii.astype(np.int16) # integer part of radii (bin size = 1)
    # The particularly tricky part, must average values within each radii bin
    # Start by looking for where radii change values
    deltar = pix_rad_int[1:] - pix_rad_int[:-1] # assume all radii represented (more work if not)
    rind = np.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number of pixels in radius bin
    csim = np.cumsum(sorted_im, dtype=np.float64) # cumulative sum for increasing radius
    # total in one bin is simply difference between cumulative sum for adjacent bins
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radial_profile = tbin/nr # compute average for each bin
    return radial_profile

def phase_corr(a, b):
    tmp = spf.fft2(a)*spf.fft2(b).conj()
    tmp /= abs(tmp)
    return spf.ifft2(tmp)
    
def l2_norm(a):
    return sp.sqrt(sp.sum(a**2.))
    
def norm_cross_corr(a,b):
    a-=sp.mean(a)
    b-=sp.mean(b)
    
    return sp.sum(cross_corr(a,b))/(l2_norm(a)*l2_norm(b))
    
    
def cross_corr(a, b):
    tmp = spf.fft2(a)*spf.fft2(b).conj()
    return spf.ifft2(tmp)    

def gauss_smooth(arr, w):
    """
    Smooths an input array by w pixels
    """
    alpha = w/2.3548
    pix = arr.shape[0]
    ker = sp.exp(-(alpha**-2.)*sp.hypot(*sp.ogrid[-pix/2:pix/2,-pix/2:pix/2])**2.)
    return sp.real(spf.fftshift(spf.ifft2( spf.fft2(arr)*spf.fft2(ker) )))

def error_metric(det_mod, det_est):
    return sp.sum(det_mod-abs(det_est))**2./sp.sum(det_mod)**2.

def energy_to_wavelength(energy):
    """
    Converts an energy in keV to wavelength in metres.
    """
    wl = spc.physical_constants['Planck constant'][0]
    wl *= spc.speed_of_light
    wl /= (energy*1e3*spc.elementary_charge)
    
    return wl

def phase(arr):
    return sp.arctan2(arr.imag, arr.real)

def pad_shift(arr1, nx, ny):
    """
    Shifts an array by nx and ny respectively.
    """
    initial_size = arr1.shape
    new_arr = sp.zeros((2*arr1.shape[0], 2*arr1.shape[1]))
    new_arr[:arr1.shape[0], :arr1.shape[1]] = arr1
    arr1=new_arr
    pix = arr1.shape[0]
    nx*=1.
    ny*=1.
    window = spf.fftshift(np.hamming(pix)[:,np.newaxis]*np.hamming(pix)[np.newaxis,:])
    if ((nx % 1. == 0.) and (ny % 1. ==0)):
        return sp.roll(sp.roll(arr1, int(ny), axis=0),
                       int(nx), axis=1 )
    else:
        freqs = spf.fftfreq(pix)
        phaseFactor = sp.zeros((pix,pix),dtype=complex)
        for i in xrange(pix):
            for j in xrange(pix):
                phaseFactor[i,j] = sp.exp(complex(0., -2.*sp.pi)*(nx*freqs[j]+ny*freqs[i]))
        tmp = spf.ifft2(spf.fft2(arr1)*phaseFactor*window)

        return sp.real(tmp.copy())
    
def shift(arr1, nx, ny):
    """
    Shifts an array by nx and ny respectively.
    """
    
    pix = arr1.shape[0]
    nx*=1.
    ny*=1.
    
    if ((nx % 1. == 0.) and (ny % 1. ==0)):
        return sp.roll(sp.roll(arr1, int(ny), axis=0),
                       int(nx), axis=1 )
    else:
        freqs = spf.fftfreq(pix)
        phaseFactor = sp.zeros((pix,pix),dtype=complex)
        for i in xrange(pix):
            for j in xrange(pix):
                phaseFactor[i,j] = sp.exp(complex(0., -2.*sp.pi)*(nx*freqs[j]+ny*freqs[i]))
        tmp = spf.ifft2(spf.fft2(arr1)*phaseFactor)
        return sp.real(tmp.copy())
        
def cshift(arr1, nx, ny):
    """
    Shifts a complex array by nx and ny respectively.
    """
    nx*=1.
    ny*=1.
    
    if ((nx % 1. == 0.) and (ny % 1. ==0)):
        return sp.roll(sp.roll(arr1, int(ny), axis=0),
                       int(nx), axis=1 )
    else:
    
        return spf.ifft2(spnf.fourier_shift(spf.fft2(arr1),(ny,nx)))
    
def fftw_cshift(arr1, nx, ny, fft, ifft):
    
    """
    Shifts an array by nx and ny respectively.
    """
    
    pix = arr1.shape[0]
    nx*=1.
    ny*=1.
    
    if ((nx % 1. == 0.) and (ny % 1. ==0)):
        return sp.roll(sp.roll(arr1, int(ny), axis=0),
                       int(nx), axis=1 )
    else:
        freqs = spf.fftfreq(pix)
        arr1=fft(arr1)
        for i in xrange(pix):
            for j in xrange(pix):
                arr1[i,j] *= sp.exp(complex(0., -2.*sp.pi)*(nx*freqs[i]+ny*freqs[j]))
        arr1 = ifft(arr1)
        return arr1
    
        
def shift_nsa(arr1, nx, ny):
    """
    Shifts an array by nx and ny respectively.
    """
    
    pix_x, pix_y = arr1.shape[0], arr1.shape[1]
    nx*=1.
    ny*=1.
    
    if ((nx % 1. == 0.) and (ny % 1. ==0)):
        return sp.roll(sp.roll(arr1, int(ny), axis=0),
                       int(nx), axis=1 )
    else:
        freqs_x, freqs_y = spf.fftfreq(pix_x), spf.fftfreq(pix_y)
        phaseFactor = sp.zeros((pix_x,pix_y),dtype=complex)
        for i in xrange(pix_x):
            for j in xrange(pix_y):
                phaseFactor[i,j] = sp.exp(complex(0., -2.*sp.pi)*(nx*freqs_x[i]+ny*freqs_y[j]))
        tmp = spf.ifft2(spf.fft2(arr1)*phaseFactor)
        return sp.real(tmp.copy())
       
def zp_stats(energy, delta_r, r):
    l = energy_to_wavelength(energy)
    focal_length = 2*r*delta_r/l
    NA = l/(2*delta_r)
    DOF = l/(2*NA**2.)
    axial_res = 2*l/NA**2.
    lateral_res = l/(2.*NA)
    print 'focal length: %1.2e\nNA: %1.2e\nDOF: %1.2e\nAxial resolution: %1.2e\nLateral resolution: %1.2e' % (focal_length, NA, DOF, axial_res, lateral_res)
    
def aperture_stats(energy, z, x):
    l=energy_to_wavelength(energy)
    NA = sp.sin(sp.arctan(x/z))
    axial_res = 2*l/NA**2.
    lateral_res = l/(2.*NA)
    print 'NA: %1.2e\nAxial resolution: %1.2e\nLateral resolution: %1.2e' % (NA, axial_res, lateral_res)    
    
def ccd_stats(energy, npix, pix_size, z_sam_det):
    NA = sp.sin(sp.arctan(0.5*npix*pix_size/z_sam_det))
    l = energy_to_wavelength(energy)
    axial_res = 2*l/NA**2.
    lateral_res = l/(2.*NA)
    
    print 'NA: %1.2e\nAxial resolution: %1.2e\nLateral resolution: %1.2e' % (NA, axial_res, lateral_res)

def cdi_info(energy, h, z, pix, del_x_d, verbose = False):
    """
    h - object size\nz - sam-det dist\npix - # of pix\ndel_x_d - pixel size
    """
    x = (pix/2.)*del_x_d
    l = energy_to_wavelength(energy)
    NF = lambda nh, nl, nz : nh**2./(nl*nz)
    del_x_s = lambda l, z, x : (l*z)/(2.*x)
    nNF = NF(h,l,z)
    OS = lambda l,z,x,h,pix : ((pix*del_x_s(l,z,x))**2.)/(h**2.)
    nOS = OS(l,z,x,h,pix)
    if verbose:
        pyl.figure()
        zrange = sp.linspace(0, 2*z, 100)
        pyl.plot(zrange, sp.log(NF(h,l,zrange)))
        pyl.title('NF')
        pyl.xlabel('z [m]')
        pyl.ylabel('log NF')
        pyl.figure()
        pyl.plot(zrange, sp.log(OS(l,zrange, x, h, pix)))
        pyl.title('OS')
        pyl.xlabel('z [m]')
        pyl.ylabel('log OS')
    
    print 'NF: %1.2e\nOS: %1.2e\ndel_x_d: %1.2e\nw_d: %1.2e\ndel_x_s: %1.2e\nw_s: %1.2e' % (nNF, nOS, del_x_d, pix*del_x_d, del_x_s(l,z,x), del_x_s(l,z,x)*pix)
    aperture_stats(energy, z, x)
    

import numpy as n
import scipy.interpolate
import scipy.ndimage

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [n.float64, n.float32]:
        a = n.cast[float](a)

    m1 = n.cast[int](minusone)
    ofs = n.cast[int](centre) * 0.5
    old = n.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = n.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = n.indices(newdims)[i]
            dimlist.append( (oextensionsld[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = n.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = n.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [n.arange(i, dtype = n.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = n.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = n.mgrid[nslices]

        newcoords_dims = range(n.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (n.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported."
        return None

    
import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

def readNetCDF(filename, varName='intensity'):
    """
    Reads a netCDF file and returns the varName variable.
    """ 
    import Scientific
    import Scientific.IO
    import Scientific.IO.NetCDF
    ncfile = Scientific.IO.NetCDF.NetCDFFile(filename,"r")
    var1 = ncfile.variables[varName]
    data = sp.array(var1.getValue(),dtype=float)
    ncfile.close()
    return data

def getNetCDF_variables(filename):
    ncfile = Scientific.IO.NetCDF.NetCDFFile(filename,'r')
    vars = ncfile.variables.keys()
    ncfile.close()
    return vars

def showNetCDF(filename):
    
    """
    matshows the netcdf in filename
    """
    
    img=readNetCDF(filename,'intensity')
    pyl.matshow(img)
    
def writeNetCDF(filename, varName, arr):
    """
    Creates a netCDF file with arr stored in variable varName
    """

    pix = arr.shape[0]
    ncfile = Scientific.IO.NetCDF.NetCDFFile(filename, "a")
    ncfile.createDimension("pixdim",2048)
    var = ncfile.createVariable(varName, "d", ("pixdim","pixdim") )
    var.assignValue(arr.copy().astype(float))
    ncfile.close()

def from_colour_image(fn):
    import Image
    import ImageOps
    return sp.misc.fromimage(ImageOps.grayscale(Image.open(fn))).astype(float)
    
def zp_focal_length(radius, outermost_zone_width, wl):
    return 2.0*radius*outermost_zone_width/wl
    
def avim(dtype = float, pix=2048):
    '''
    averages scipy stored binary DATs.
    Inputs: the dtype and the image size
    '''
    filenames = tkFileDialog.askopenfilenames(title = 'select files to average',
                filetypes=[('Scipy DATs','.dat')])
    im = sp.zeros((pix,pix))
    for file in filenames:
        im += 1.*sp.fromfile(file,dtype).reshape(pix,pix)
    return im/len(filenames)
    
def res(z,energy,pix_size,n_pixels):
    return (z*energy_to_wavelength(energy))/(n_pixels*pix_size)
def fov(z,energy,pix_size):
    return (z*energy_to_wavelength(energy))/pix_size
def NF(h,energy,z):
    return h**2./(energy_to_wavelength(energy)*z)

def main():
    
    return 0

if __name__ == '__main__': main()
