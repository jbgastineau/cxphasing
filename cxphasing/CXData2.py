"""
.. module:: CXData2.py
   :platform: Unix
   :synopsis: A class for coherent X-ray phasing data.

.. moduleauthor:: David Vine <djvine@gmail.com>


"""

import scipy as sp
import numpy as np
import scipy.fftpack as spf
import scipy.ndimage as spn
from numpy.random import uniform
from numpy import pad
import os
import pdb
import pylab
import shutil
import sys
import operator
from round_scan import round_roi


import glob
import multiprocessing as mp
import time
from matplotlib import cm
from images2gif import writeGif
from CXFileReader import CXFileReader
from cxparams import CXParams as CXP

debug = True

def fft2(x):
    # Wrapped for fft2 that handles CXData objects and ndarrays
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(spf.fft2(x.data[i]))
        return CXData(data=l)
    elif isinstance(x, CXModal):
        l=[]
        for mode in range(len(x.modes)):
            l.append(fft2(x.modes[mode]))
        return CXModal(modes=l)
    elif isinstance(x, np.ndarray):
        return spf.fft2(x)
    else:
        raise Exception('Unknown data type passed to fft2')


def ifft2(x):
    # Wrapped for ifft2 that handles CXData objects
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(spf.ifft2(x.data[i]))
        return CXData(data=l)
    elif isinstance(x, CXModal):
        l=[]
        for mode in range(len(x.modes)):
            l.append(ifft2(x.modes[mode]))
        return CXModal(modes=l)
    elif isinstance(x, np.ndarray):
        return spf.ifft2(x)
    else:
        raise Exception('Unknown data type passed to ifft2')


def fftshift(x):
    # Wrapper for fftshift that handles CXData objects
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(spf.fftshift(x.data[i]))

        return CXData(data=l)
    elif isinstance(x, np.ndarray):
        return spf.fftshift(x)
    else:
        raise Exception('Unknown data type passed to fftshift')


def abs(x):
    # Wrapper for abs that handles CXData objects
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(np.abs(x.data[i]))
        return CXData(data=l)
    elif isinstance(x, CXModal):
        l=[]
        for mode in range(len(x.modes)):
            l.append(abs(x.modes[mode]))
        return CXModal(modes=l)
    elif isinstance(x, np.ndarray):
        return np.abs(x)
    else:
        raise Exception('Unknown data type passed to abs')


def angle(x):
    # Wrapper for angle that handles CXData objects
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(sp.angle(x.data[i]))

        return CXData(data=l)
    elif isinstance(x, np.ndarray):
        return sp.angle(x)
    else:
        raise Exception('Unknown data type passed to angle')


def exp(x):
# Wrapper for exp that handles CXData objects
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(sp.exp(x.data[i]))

        return CXData(data=l)
    elif isinstance(x, np.ndarray):
        return sp.exp(x)
    else:
        raise Exception('Unknown data type passed to exp')


def log(x):
    # Wrapper for exp that handles CXData objects
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(sp.log(x.data[i]))

        return CXData(data=l)
    elif isinstance(x, np.ndarray):
        return sp.log(x)
    else:
        raise Exception('Unknown data type passed to log')


def conj(x):
    """
    Wrapper for conjugate on a CXData object.
    """
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(sp.conj(x.data[i]))
        return CXData(data=l)
    elif isinstance(x, CXModal):
        l=[]
        for mode in range(len(x.modes)):
            l.append(conj(x.modes[mode]))
        return CXModal(modes=l)
    elif isinstance(x, np.ndarray):
        return sp.conj(x)
    else:
        raise Exception('Unknown data type passed to conj')

def sqrt(x):
    """
    Wrapper for square root on a CXData object.
    """
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(sp.sqrt(x.data[i]))
        return CXData(data=l)
    elif isinstance(x, CXModal):
        l=[]
        for mode in range(len(x.modes)):
            l.append(exp(x.modes[mode]))
        return CXModal(modes=l)
    elif isinstance(x, np.ndarray):
        return sp.sqrt(x)
    else:
        raise Exception('Unknown data type passed to conj')


def sum(x):
    """
    Sum over arrays.
    """

    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            if i==0:
                l.append(x.data[0])
            else:
                l[0] += x.data[i]
        return CXData(data=l)
    elif isinstance(x, CXModal):
        l=[]
        for mode in range(len(self.modes)):
            l.append(sum(self.modes[mode]))
        return CXModal(modes=l)
    elif isinstance(x, np.ndarray):
        return sp.sum(x)
    else:
        raise Exception('Unknown data type pass to sum')


def worker(func):

    def worker2(self=None, *args, **kwargs):
        try:
            kwargs['no_decorate']
            return func(self, args[0], args[1], args[2], args[3], args[4], args[5])
        except KeyError:
            cnt = 0
            jobs, results = args[0], args[1]
            while True:
                job_args = jobs.get()
                if job_args[0]==None:  # Deal with Poison Pill
                    print '{}: Exiting. {:d} jobs completed.'.format(mp.current_process().name, cnt)
                    jobs.task_done()
                    break
                if job_args[0]%np.floor(job_args[1]/10)==0:
                    print 'Processed {:d} out of {:d} files.'.format(job_args[0], job_args[1])

                res = func(self, *job_args)
                cnt+=1
                jobs.task_done()
                results.put(res)
            return worker2
    return worker2


class CXData(CXFileReader):

    """
    Defines a class for holding and interacting with coherent x-ray data.

    ...

    Attributes
    ----------
    data: list
        list of complex arrays that hold all of the phase retrieval data.
    name: str
        name of instance. Used for logging.
    savepath: str 
        location where this data should be saved.



    Methods
    -------


    """

    def __init__(self, *args, **kwargs):

        self.data = None
        self.savepath = None

        for kw in kwargs:
            # Data attribute must be a list of arrays
            if kw=='data':
                if isinstance(kwargs['data'], list):
                    self.data = kwargs['data']
                elif isinstance(kwargs['data'], np.ndarray):
                    self.data = [kwargs['data']]
            else:
                setattr(self, kw, kwargs[kw])

    def __repr__(self):
        try:
            s=repr(self.data[0])
        except:
            s=''
        try:
            return '<{} at {}>\n{} arrays ({:d}x{:d}px).\n{}'.format(self.__class__,
                hex(id(self)), len(self.data), self.data[0].shape[0], self.data[0].shape[1], s)
        except AttributeError:
            return '<{} at {}>\nNo data attribute present.'.format(self.__class__, hex(id(self)))

    def __add__(self, other):
        if isinstance(other, CXData):
            l=[]
            for i in xrange(len(self.data)):
                l.append(self.data[i]+other.data[i])
            return CXData(data=l)
        elif isinstance(other, (int, float, complex)):
            l=[]
            for i in xrange(len(self.data)):
                l.append(self.data[i]+other)
            return CXData(data=l)

    def __iadd__(self, other):
        if isinstance(other, CXData):
            for i in xrange(len(self.data)):
                self.data[i]+=other.data[i]
            return self
        elif isinstance(other, (int, float, complex)):
            for i in xrange(len(self.data)):
                self.data[i]+=other
            return self

    def __sub__(self, other):
        if isinstance(other, CXData):
            l=[]
            for i in xrange(len(self.data)):
                l.append(self.data[i]-other.data[i])
            return CXData(data=l)
        if isinstance(other, (int, float, complex)):
            l=[]
            for i in xrange(len(self.data)):
                l.append(self.data[i]-other)
            return CXData(data=l)

    def __isub__(self, other):
        if isinstance(other, CXData):
            for i in xrange(len(self.data)):
                self.data[i]-=other.data[i]
            return self
        if isinstance(other, (int, float, complex)):
            for i in xrange(len(self.data)):
                self.data[i]-=other.data
            return self

    def __pow__(self, power):
        l=[]
        for i in xrange(len(self.data)):
            l.append(self.data[i]**power)
        return CXData(data=l)

    def __mul__(self, other):
        if isinstance(other, CXData):
            l=[]
            for i in xrange(len(self.data)):
                l.append(self.data[i]*other.data[i])
            return CXData(data=l)
        elif isinstance(other, (int, float, complex)):
            l=[]
            for i in xrange(len(self.data)):
                l.append(self.data[i]*other)
            return CXData(data=l)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, CXData):
            for i in xrange(len(self.data)):
                self.data[i]*=other.data[i]
            return self
        elif isinstance(other, (int, float, complex)):
            for i in xrange(len(self.data)):
                self.data[i]*=other
            return self

    def __div__(self, other):
        if isinstance(other, CXData):
            l=[]
            for i in xrange(len(self.data)):
                l.append(self.data[i]/other.data[i])
            return CXData(data=l)
        elif isinstance(other, (int, float, complex)):
            l=[]
            for i in xrange(len(self.data)):
                l.append(self.data[i]/other)
            return CXData(data=l)

    def __rdiv__(self, other):
        return self.__mul__(other)

    def __idiv__(self, other):
        if isinstance(other, CXData):
            for i in xrange(len(self.data)):
                self.data[i]/=other.data[i]
            return self
        elif isinstance(other, (int, float, complex)):
            for i in xrange(len(self.data)):
                self.data[i]/=other
            return self

    def __len__(self):
        return len(self.data)

    def __del__(self):
        # Remove this instance from the CXData __all__ variable
        try:
            print 'Deleting {}'.format(self.kwargs['itype'])
            CXData.__all__.pop(self.kwargs['itype'])
        except (AttributeError, KeyError):
            pass

    def __getitem__(self, s):
        """
        Allows extracting a subarray from self.data or a single array from a list of arrays.

        Implements subpixel shifting for seamless indexing of a fractional number of pixels.

        The returned array must be an integer number of pixels.

        E.g a[0:100.6] doesn't make any sense
        but a[0.6:100.6] does.

        a[0] is equivalent to a.data[0]


        """
        if isinstance(s, int):
            return CXData(data=self.data[s])
        else:
            y, x = s
            xstart = x.start or 0
            xstop = x.stop or self.data[0].shape[0]-1
            ystart = y.start or 0
            ystop = y.stop or self.data[0].shape[1]-1

            dx, dy = -np.mod(xstart, 1), -np.mod(ystart, 1)

            l = []
            for data in self.data:
                l.append(self.shift(data[xstart // 1:xstop // 1, ystart //1: ystop //1], dx, dy))

            return CXData(data=l)

    def __setitem__(self, s, arr):

        """
        Embed a smaller array in a larger array.

        a[s] = arr
        """
        if isinstance(s, int):
            if len(arr)>1:
                raise Exception('Cannot set single array with list of arrays.')
            self.data[s]=arr.data[0]
        else:

            y, x = s
            xstart = x.start or 0
            xstop = x.stop or self.data[0].shape[0]-1
            ystart = y.start or 0
            ystop = y.stop or self.data[0].shape[1]-1
            dx, dy = np.mod(xstart, 1), np.mod(ystart, 1)

            l=[]
            if isinstance(arr, CXData):
                for i, data in enumerate(self.data):
                    l.append(data.copy())
                    l[i][xstart // 1:xstop // 1, ystart //1: ystop //1] = self.shift(arr.data[i], dx, dy)
                self.data = l
            elif isinstance(arr, np.ndarray):
                for i, data in enumerate(self.data):
                    l.append(data.copy())
                    l[i][xstart // 1:xstop // 1, ystart //1:ystop //1] = self.shift(arr, dx, dy)
                self.data = l
            elif isinstance(arr, (int, float)):
                for i, data in enumerate(self.data):
                    l.append(data.copy())
                    l[i][xstart // 1:xstop // 1, ystart //1: ystop //1] = arr
                    l[i] = self.shift(l[i], dx, dy)
                self.data = l

    @staticmethod
    def inner_product(u, v):
        return sp.sum((conj(u)*v).data[0])/(u.data[0].shape[0]*u.data[0].shape[1])

    @staticmethod
    def proj_u_v(u, v):
        return u*(CXData.inner_product(v, u)/CXData.inner_product(u, u))

    def max(self):
        """
        Return a list of maximum (absolute) value(s) of (complex) array(s).
        """
        if len(self.data)==1:
            return abs(self.data[0]).max()
        else:
            return [abs(element).max() for element in self.data]

    def min(self):
        """
        Return a list of minimum (absolute) value(s) of (complex) array(s).
        """
        if len(self.data)==1:
            return abs(self.data[0]).min()
        else:
            return [abs(element).min() for element in self.data]

    def normalise(self, val=1.):
        """
        Rebase data from 0 to 1.
        """
        if CXP.reconstruction.verbose:
            CXP.log.info('Rebasing data from 0 to {:3.2f}'.format(val))

        for i in xrange(len(self.data)):
            self.data[i] -= abs(self.data[i]).min()
            self.data[i] /= abs(self.data[i]).max()
            self.data[i] *= val

    def append(self, other):
        if isinstance(other, CXData):
            for data in other.data:
                self.data.append(data)
        elif isinstance(other, np.ndarray):
            self.data.append(other)

    def square_root(self):
        if CXP.reconstruction.verbose:
            CXP.log.info('Taking square root.')
        for i in xrange(len(self.data)):
            self.data[i] = pow(self.data[i], 0.5)

    def fft_shift(self):
        if CXP.reconstruction.verbose:
            CXP.log.info('Performing FFT shift.')
        for i in xrange(len(self.data)):
            self.data[i] = spf.fftshift(self.data[i])

    def len(self):
        return len(self.data)

    @staticmethod
    def shift_inner(arr, nx, ny, window=False, padding='reflect'):
        """
        Shifts an array by nx and ny respectively.

        """

        if ((nx % 1. == 0.) and (ny % 1. ==0)):
            return sp.roll(sp.roll(arr, int(ny), axis=0),
                           int(nx), axis=1)
        else:
            atype = arr.dtype
            if padding:
                x, y = arr.shape
                pwx, pwy = int(pow(2., np.ceil(np.log2(1.5*arr.shape[0])))), int(pow(2., np.ceil(np.log2(1.5*arr.shape[1]))))
                pwx2, pwy2 = (pwx-x)/2, (pwy-y)/2
                if pad=='zero':
                    arr = pad.with_constant(arr, pad_width=((pwx2, pwx2), (pwy2, pwy2)))
                else:
                    arr = pad.with_reflect(arr, pad_width=((pwx2, pwx2), (pwy2, pwy2)))
            phaseFactor = sp.exp(complex(0., -2.*sp.pi)*(ny*spf.fftfreq(arr.shape[0])[:, np.newaxis]+nx*spf.fftfreq(arr.shape[1])[np.newaxis, :]))
            if window:
                window = spf.fftshift(CXData._tukeywin(arr.shape[0], alpha=0.35))
                arr = spf.ifft2(spf.fft2(arr)*phaseFactor*window)
            else:
                arr = spf.ifft2(spf.fft2(arr)*phaseFactor)
            if padding:
                arr = arr[pwx/4:3*pwx/4, pwy/4:3*pwy/4]

        if atype == 'complex':
            return arr
        else:
            return np.real(arr)

    @staticmethod
    def shift(x, nx, ny, **kwargs):
        if isinstance(x, CXData):
            l=[]
            for data in x.data:
                l.append(CXData.shift_inner(data.copy(), nx, ny, **kwargs))
            return CXData(data=l)

        elif isinstance(x, np.ndarray):
            return CXData.shift_inner(x, nx, ny)

    def ishift(self, nx, ny, **kwargs):
        # Inplace version of shift
        l=[]
        for data in self.data:
            for data in self.data:
                l.append(self.shift_inner(data.copy(), nx, ny, kwargs))
        self.data = l
        return self

    def rot90(self, i):
        # Rotate by 90 degrees i times
        if CXP.reconstruction.verbose:
            CXP.log.info('Rotating data by {:d}'.format(i*90))
        for j, data in enumerate(self.data):
            self.data[j] = sp.rot90(data, i)

    def find_dead_pixels(self):
        # Return coordinates of pixels with a standard deviation of zero

        dead_pix = sp.where(abs(np.std(self.data, axis=0))<machine_precision)
        if CXP.reconstruction.verbose:
            CXP.log.info('Found {0:d} dead pixels'.format(len(dead_pix)))
        return dead_pix

    def zero_dead_pixels(self):
        if CXP.reconstruction.verbose:
            CXP.log.info('Setting dead pixels to zero')
        self.data[self.find_dead_pixels()]=0.

    def threshhold(self, threshhold=None):
        if not threshhold:
            threshhold = CXP.preprocessing.threshhold_raw_data
        if CXP.reconstruction.verbose:
            CXP.log.info('Applying threshhold to data at {:3.2f} and rebasing to 0.'.format(threshhold))

        for i, data in enumerate(self.data):
            tdata = sp.where(data<threshhold, threshhold, data)
            tdata-=tdata.min()
            self.data[i]=tdata

    def symmetrize_array_shape(self, qxqy0=None, desired_shape=None):

        x0, y0 = self.data[0].shape
        if desired_shape is None:
            desired_shape = CXP.preprocessing.desired_array_shape
        if qxqy0 is None:
            qx, qy = CXP.preprocessing.qx0qy0
        else:
            qx, qy = qxqy0
        if CXP.reconstruction.verbose:
            CXP.log.info('Symmetrizing array shape.\n\tCurrent shape:\t{}x{}\n\tNew shape:\t{}x{}\n\tCentred on:\t{},{}'.format(
                        x0, y0, desired_shape, desired_shape, qx, qy))

        # Cropping or padding?
        qx_lower, qx_upper = qx-desired_shape/2, qx+desired_shape/2
        qy_lower, qy_upper = qy-desired_shape/2, qy+desired_shape/2

        if qx_lower<0:  # Crop
            nxl, mxl = np.abs(qx_lower), 0
        else:  # Pad
            nxl, mxl = 0, qx_lower
        if qy_lower<0:  # Crop
            nyl, myl = np.abs(qy_lower), 0
        else:  # Pad
            nyl, myl = 0, qy_lower

        if qx_upper<x0:  # Crop
            nxu, mxu = desired_shape, qx+desired_shape/2
        else:  # Pad
            nxu, mxu = x0-qx_lower, x0
        if qy_upper<y0:  # Crop
            nyu, myu = desired_shape, qy+desired_shape/2
        else:  # Pad
            nyu, myu = y0-qy_lower, y0

        for i in range(len(self.data)):
            tmp = sp.zeros((desired_shape, desired_shape))
            tmp[nxl:nxu, nyl:nyu] = self.data[i][mxl:mxu, myl:myu]
            self.data[i] = tmp

        CXP.p = CXP.preprocessing.desired_array_shape

    def treat_beamstop(self):
        factor = CXP.measurement.beam_stop_factor.keys()[0]
        x0, y0 = CXP.measurement.beam_stop_factor[factor][0]
        x1, y1 = CXP.measurement.beam_stop_factor[factor][1]
        for i in range(len(self.data)):
            self.data[i][x0:x1, y0:y1]*=factor

    def save(self, path=None):

        if path:
            filepath = path
        else:
            filepath = self.savepath
        try:
            CXP.log.info('Saving {} to:\n\t{}'.format(self.name, filepath))
        except AttributeError:
            CXP.log.info('Saving to:\n\t{}'.format(filepath))
        try:
            np.savez(filepath, *self.data)
        except IOError as e:
            CXP.log.error(e)
            raise Exception('Could not save {} to {}'.format(self.kwargs['name'], path))

    def load(self, path=None):
        if path:
            filepath = path
        else:
            filepath = self.filename
        CXP.log.info('Loading data from:\n\t{}'.format(filepath))
        try:
            self.data = self.openup(filepath)
        except IOError as e:
            CXP.log.error(e)
            raise Exception('Could not load file from {}'.format(filepath))
        if not isinstance(self.data, list):
            self.data = [self.data]

    def init_data(self, *args, **kwargs):

        if args[0] == 'det_mod':
            if CXP.actions.preprocess_data:
                self.read_in_data()
            else:
                self.load()

        elif args[0] == 'probe_det_mod':
            if CXP.actions.preprocess_data:
                #  Get list of white files
                CXP.log.info('Preprocessing probe detector modulus.')
                if CXP.io.whitefield_filename not in [None, '']: # If whitefields were measured

                    wfilename, wfilerange, wn_acqs = [CXP.io.whitefield_filename, CXP.io.whitefield_filename_range,
                                                      CXP.measurement.n_acqs_whitefield]
                    self.pattern = wfilename.count('{')
                    if self.pattern == 1:
                        wf = [wfilename.format(i) for i in range(wfilerange[0], wfilerange[1])]
                    elif self.pattern == 2:
                        wf = [wfilename.format(wfilerange[0], i) for i in range(wn_acqs)]
                    elif self.pattern == 3:
                        wf = glob.glob(wfilename.split('}')[0]+'}*')

                    res = self.preprocess_data_stack(0, 1, wf, self.pattern, None, None, no_decorate=True)
                    self.data = res[1]
                else: #Guesstimate the whitefield from the average of the diffraction patterns
                    pass
            else:
                self.load(CXData.raw_data_filename_string.format('probe_det_mod'))

            try:
                probe = self.__class__.__all__['probe']
                probe.data[0] = spf.ifft2(self.data[0]*exp(complex(0., 1.)*sp.angle(spf.fft2(probe.data[0]))))
                CXP.log.info('Applied probe modulus constraint.')
            except (AttributeError, KeyError):
                pass

        elif args[0] == 'dark':
            if CXP.actions.preprocess_data:
                # Get list of dark files
                CXP.log.info('Preprocessing darkfield.')
                dfilename, dfilerange, dn_acqs = [CXP.io.darkfield_filename, CXP.io.darkfield_filename_range,
                                                  CXP.measurement.n_acqs_darkfield]
                self.pattern = dfilename.count('{')
                if self.pattern == 1:
                    df = [dfilename.format(i) for i in range(dfilerange[0], dfilerange[1])]
                elif self.pattern == 2:
                    df = [dfilename.format(dfilerange[0], i) for i in range(dn_acqs)]
                elif self.pattern == 3:
                    df = glob.glob(dfilename.split('}')[0]+'}*')
                res = self.preprocess_data_stack(0, 1, df, self.pattern, None, None, no_decorate=True)
                self.data = res[1]
            else:
                self.load(CXData.raw_data_filename_string.format('probe_det_mod'))


    def read_in_data(self):

        self.completed_filenames = []  # Keep track of what's been processed already for online analysis
        self.job_filenames = []  # Bundle stack of images for preprocessing
        self.pattern = None

        # Determine which files to read in
        CXP.log.info('Reading in & preprocessing raw data...')

        #Pattern 1: 'image_{:d}.xxx'
        #Pattern 2: 'image_{:d}_{:d}.xxx'
        #Pattern 3: 'image_{:d}_{:d}_{val}.xxx'

        if self.pattern is None:  # Pattern is not yet dertermined

            filename, filerange, n_acqs = [CXP.io.data_filename, CXP.io.data_filename_range, CXP.measurement.n_acqs_data]

            self.pattern = filename.count('{')
            CXP.log.info('Detected filename pattern: {:d}'.format(self.pattern))
            if self.pattern == 0:
                raise Exception('NamingConventionError:\nPlease read CXParams for more info on file naming conventions.')

        try:
            n0, n1 = filerange[0], filerange[1]+1
        except IndexError:
            n0 = n1 = filerange[0]

        if CXP.io.darkfield_filename is not '':  # dark
            try:
                dark = self.__class__.__all__['dark']
                CXP.log.info('Found darkfield.')
            except KeyError:
                dark = CXData(itype='dark')
                dark.save()
        else:
            CXP.log.info('Not processing darkfields.')
            dark = None

        if CXP.io.whitefield_filename is not '':  # white
            try:
                probe_det_mod = self.__class__.__all__['probe_det_mod']
                CXP.log.info('Found probe detector modulus.')
            except KeyError:
                probe_det_mod = CXData(itype='probe_det_mod')
                probe_det_mod.save()
        else:
            CXP.log.info('Not processing whitefields.')
            probe_det_mod = None

        old_verbosity = CXP.reconstruction.verbose
        CXP.reconstruction.verbose = False

        jobs = mp.JoinableQueue()
        results = mp.Queue()

        n_processes = mp.cpu_count()

        then = time.time()
        cnt=0
        missing_frames = False
        l=[]
        CXP.log.info('Dividing raw data into jobs over {:d} processes.'.format(n_processes))

        for i in range(n0, n1):
            if self.pattern == 1:
                s = [filename.format(i)]
            else:
                s = glob.glob((filename.split('}')[0]+'}*').format(i))
            # Include only files that haven't been processed yet
            # s = [fn for fn in s if fn not in self.completed_filenames]
            if len(s)==0:
                CXP.log.error('Globbed 0 files in CXData@read_in_files')
                sys.exit(1)
            if self.pattern==1:
                try:
                    s=s[0]
                    self.completed_filenames.append(s)
                    if cnt<n_acqs:
                        l.append(s)
                        cnt+=1
                    if cnt>=n_acqs:
                        self.job_filenames.append(l)
                        cnt=0
                        l=[]
                except IndexError:
                    missing_frames = True
                    CXP.log.error('Missing frame: {:s}'.format(filename.format(i)))
            else:
                self.completed_filenames+=s
                self.job_filenames.append(s)
        if missing_frames:
            print "There were missing frames. Choose 'c' to continue or 'q' to quit."
            pdb.set_trace()
        p = [mp.Process(target=self.preprocess_data_stack, args=(jobs, results))
            for i in range(n_processes)]

        for process in p:
            process.start()

        n_jobs = len(self.job_filenames)

        for i in range(n_jobs):
            jobs.put((i, n_jobs, self.job_filenames[i], self.pattern, probe_det_mod, dark))

        # Add Poison Pill
        for i in range(n_processes):
            jobs.put((None, None, None, None, None, None))

        CXP.log.info('{:3.2f} seconds elapsed dividing jobs between processes.'.format(time.time()-then))
        then = time.time()
        cnt = 0
        self.data = [None]*n_jobs
        while True:
            if not results.empty():
                i, data = results.get()
                self.data[i] = data[0]
                cnt+=1
            elif cnt==n_jobs:
                break

        jobs.join()

        jobs.close()
        results.close()

        for process in p:
            process.join()

        CXP.log.info('{:3.2f} seconds elapsed preprocessing data.'.format(time.time()-then))
        CXP.reconstruction.verbose = old_verbosity

        #self._sequence_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'sequences'])
        #self._cur_sequence_dir = self._sequence_dir+'/sequence_{:d}'.format(CXP.reconstruction.sequence)
        #self.save(path=self._cur_sequence_dir+'/det_mod.npy')

    @worker
    def preprocess_data_stack(self, stack_num, n_jobs, file_list, pattern, white, dark):
        # Average, merge and preprocess a stack of images
        # Typically a stack corresponds to one ptychographic position
        l=[]
        tmp=None
        # First - average according to the pattern
        if pattern in [1, 2]:
            # Averaging only
            for filename in file_list:
                if tmp is None:
                    tmp = self.openup(filename)
                else:
                    tmp += self.openup(filename)
            l.append(tmp/len(file_list))
        elif pattern == 3:
            # Average then merge
            d={}
            unique_times = list(set([t.split('_')[3] for t in file_list]))
            for filename in file_list:
                t = filename.split('.')[0].split('_')[-1]
                if t not in d.keys():
                    d[t] = (1, self.openup(filename))
                else:
                    d[t][0] += 1
                    d[t][1] += self.openup(filename)

            for key, (i, val) in d.iteritems():
                val /= i

            # Check for saturated values and merge variable exposure times
            max_time = max(unique_times)
            if CXP.preprocessing.saturation_level>0:
                for key in d.keys():
                    wh = sp.where(d[key]>=CXP.preprocessing.saturation_level)
                    d[key][wh] = 0
                    if tmp == 0:
                        tmp = d[key] * max_time/float(key)
                    else:
                        tmp += d[key] * max_time/float(key)

            l.append(tmp)

        else:
            raise Exception('NamingConventionError')

        # Do preprocessing

        data = CXData()
        data.data = l

        if CXP.measurement.beam_stop:
            data.treat_beamstop()

        data.symmetrize_array_shape()

        # CCD Specific Preprocessing
        if CXP.preprocessing.detector_type == 'ccd':

            try:
                # Dark field correction
                if dark is not None:
                    print('Dark field correcting data')
                    data-=dark

                # Dark correct white field
                if white is not None:
                    print('Dark field correcting whitefield')
                    white-=dark

            except UnboundLocalError:
                print('No darkfield subtraction performed.')

        # PAD Specific Preprocessing
        elif CXP.preprocessing.detector_type == 'pad':
            pass

        # Threshhold data
        if CXP.preprocessing.threshhold_raw_data > 0:
            data.threshhold()
            if white is not None:
                white.threshhold()

        # Bin data
        if CXP.preprocessing.bin > 1:
            data.bin()
            if white is not None:
                white.bin()

        if CXP.preprocessing.rot90!=0:
            data.rot90(CXP.preprocessing.rot90)
            if white is not None:
                white.rot90(CXP.preprocessing.rot90)

        # Take square root
        data.square_root()
        if white is not None:
            white.square_root()

        # Put in FFT shifted
        data.fft_shift()
        if white is not None:
            white.fft_shift()

        return (stack_num, data.data)

    def bin(self, n=None):

        """
        Bin a square array by grouping nxn pixels.
        Array size must be a multiple of n.

        """
        if n is None:
            n=CXP.preprocessing.bin
            # Now the detector pixel size has changed so we should update that
            CXP.experiment.dx_d *= n
            CXP.log.info('After binning new detector pixel size: {2.2e}'.format(CXP.experiment.dx_d))

        nx, ny = self.data[0].shape[0], self.data[0].shape[1]
        if not nx==ny:
            raise Exception('Array to be binned must be square')

        if not sp.mod(nx, n)==0.:
            raise Exception('Array size must be a multiple of binning factor')

        if n>nx:
            raise Exception('Binning factor must be smaller than array size')

        nn = nx/n
        l = []
        for i in xrange(len(self.data)):
            tmp = sp.zeros((nn, nn))
            for p in xrange(nn):
                for q in xrange(nn):
                    tmp[p, q] = sp.sum(self.data[i][p*n:(p+1)*n, q*n:(q+1)*n])
            l.append(tmp)

        self.data=l

    def show(self, i=0, phase=False, log=False):
        if phase:

            pylab.matshow(angle(self.data[i]), cmap=cm.hsv)
        else:
            if log:
                pylab.matshow(sp.log10(abs(self.data[i])))
            else:
                pylab.matshow(abs(self.data[i]))
        pylab.colorbar()
        pylab.show()

    def plot(self, i=0, phase=False):
        pylab.figure()
        if phase:
            pylab.plot(np.angle(self.data[i][:, self.data[i].shape[0]/2]), label='Horizontal')
            pylab.plot(np.angle(self.data[i][self.data[i].shape[1]/2, :]), label='Vertical')
        else:
            pylab.plot(np.abs(self.data[i][:, self.data[i].shape[0]/2]), label='Horizontal')
            pylab.plot(np.abs(self.data[i][self.data[i].shape[1]/2, :]), label='Vertical')
        pylab.legend()


    def copy(self):
        return CXData(data=[np.copy(arr) for arr in self.data])


class CXModal(object):

    def __init__(self, *args, **kwargs):

        self.modes = []
        self.savepath = None

        for kw in kwargs:
            # Data attribute must be a list of arrays
            if kw=='modes':
                if isinstance(kwargs['modes'], list):
                    self.modes = kwargs['modes']
                elif isinstance(kwargs['modes'], CXData):
                    self.modes = [kwargs['modes']]
            else:
                setattr(self, kw, kwargs[kw])

    def __repr__(self):

        try:
            s=repr(self.modes[0].data[0])
        except:
            s=''
        try:
            return '<{} at {}>\n{:d} modes containing {:d} arrays ({:d}x{:d}px).\n{}'.format(self.__class__,
                hex(id(self)), len(self.modes), len(self.modes[0]), self.modes[0].data[0].shape[0], 
                self.modes[0].data[0].shape[1], s)
        except AttributeError:
            return '<{} at {}>\nNo modes attribute present.'.format(self.__class__, hex(id(self)))

    def __getitem__(self, s):
        return self.modes[s]

    def __setitem__(self, s, modes):
        self.modes[s] = modes

    @staticmethod
    def _addsubmuldiv(operation, this, other):  
        if isinstance(other, CXModal):
            l=[]
            for mode in xrange(len(this.modes)):
                l.append(CXData(data=[operation(this.modes[mode].data[i], other.modes[mode].data[i]) for i in range(len(this.modes[mode].data))]))
            return CXModal(modes=l)
        elif isinstance(other, CXData):
            l=[]
            for mode in xrange(len(this.modes)):
                l.append(CXData(data=[operation(this.modes[mode].data[i], other.data[i]) for i in range(len(this.modes[mode].data))]))
            return CXModal(modes=l)
        elif isinstance(other, (int, float, complex)):
            l=[]
            for mode in xrange(len(this.modes)):
                l.append(CXData(data=[operation(this.modes[mode].data[i], other) for i in range(len(this.modes[mode].data))]))
            return CXModal(modes=l)

    @staticmethod
    def _iaddsubmuldiv(operation, this, other):
        if isinstance(other, CXModal):
            for mode in xrange(len(this.modes)):
                for i in range(len(this.modes[mode])):
                    this.modes[mode].data[i]=operation(this.modes[mode].data[i], other.modes[mode].data[i])
            return this
        elif isinstance(other, CXData):
            for mode in xrange(len(this.modes)):
                for i in range(len(this.modes[mode])):
                    this.modes[mode].data[i] = operation(this.modes[mode].data[i], other.data[i])
            return this
        elif isinstance(other, (int, float, complex)):
            for mode in xrange(len(this.modes)):
                for i in range(len(this.modes[mode])):
                    this.modes[mode].data[i] = operation(this.modes[mode].data[i], other)
            return this

    def __add__(self, other):
        return CXModal._addsubmuldiv(operator.add, self, other)

    def __iadd__(self, other):
        return CXModal._iaddsubmuldiv(operator.iadd, self, other)

    def __sub__(self, other):
        return CXModal._addsubmuldiv(operator.sub, self, other)

    def __isub__(self, other):
        return CXModal._iaddsubmuldiv(operator.isub, self, other)

    def __mul__(self, other):
        return CXModal._addsubmuldiv(operator.mul, self, other)

    def __rmul__(self, other):
        return CXModal._addsubmuldiv(operator.mul, self, other)

    def __imul__(self, other):
        return CXModal._addsubmuldiv(operator.imul, self, other)

    def __div__(self, other):
        return CXModal._addsubmuldiv(operator.div, self, other)

    def __rdiv__(self, other):
        return CXModal._addsubmuldiv(operator.div, self, other)

    def __idiv__(self, other):
        return CXModal._addsubmuldiv(operator.idiv, self, other)

    def __pow__(self, power):
        return CXModal(modes=[self.modes[mode]**power for mode in range(len(self.modes))])

    def __len__(self):
        return len(self.modes)

    def copy(self):
        return CXModal(modes=[self.modes[mode].copy() for mode in range(len(self))])
    
    @staticmethod
    def modal_sum(modal):
        return CXData(data=[ reduce(CXData.__add__, [ modal[mode][i] for mode in range(len(modal.modes)) ]).data[0] for i in range(len(modal[0].data))])

    def getat(self, i):
        """
        .. method::setat(self, i)

            return all modes at position i
        """
        return CXModal(modes=[self.modes[mode][i] for mode in range(len(self))])

    def setat(self, i, modal):
        """
        .. method::getat(self, i)

            set all modes at position i
        """
        for mode in range(len(self)):
            self.modes[mode][i] = modal.modes[mode][0]

    def normalise(self):
        mode_sum_max = CXModal.modal_sum(abs(self)).data[0].max()
        for mode in range(len(self)):
            self.modes[mode] /= mode_sum_max

    def orthogonalise(self):
        ortho = CXModal(modes=self[0][0].copy())
        for i in range(1, len(self)):
            tmp = self[i][0].copy()
            for j in range(i-1, -1, -1):
                tmp -= CXData.proj_u_v(ortho[j][0], self[i][0])
            ortho.modes.append(tmp)
        return CXModal(modes=ortho.modes)