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
from round_scan import round_roi
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import glob
import multiprocessing as mp
import time
from matplotlib import cm
from images2gif import writeGif
from CXFileReader import CXFileReader
from cxparams import CXParams as CXP
import vine_utils as vu

debug = True

#constant
machine_precision = 2e-14
ast_buf = '*'*50


def fft2(x):
    # Wrapped for fft2 that handles CXData objects and ndarrays
    if isinstance(x, CXData):
        l=[]
        for i in xrange(len(x)):
            l.append(spf.fft2(x.data[i]))

        return CXData(data=l)
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
    elif isinstance(x, np.ndarray):
        return sp.conj(x)
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

    __all__ = {}
    _top_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id])
    _sequence_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'sequences'])
    _cur_sequence_dir = _sequence_dir+'/sequence_{:d}'.format(CXP.reconstruction.sequence)
    _raw_data_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'raw_data'])
    _dpc_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'dpc'])
    _CXP_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, '.CXPhasing'])
    _py_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'python'])
    processed_filename_string = '/'.join([_cur_sequence_dir, '{}.npz'])
    raw_data_filename_string = '/'.join([_raw_data_dir, '{}.npz'])

    def __init__(self, *args, **kwargs):
        """

        Inputs
        ------
        itype - specifying if this instance is data, white or dark triggers actions to read in
        or generate the approriate value for self.data.

        Example Usage:
        det_mod = CXData(itype='det_mod')
        """

        self.log = CXP.log

        self.args = args

        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

            if 'itype' in kwargs.keys():
                if kwargs['itype'] in ['det_mod', 'probe_det_mod', 'dark']:
                    self.filename = CXData.raw_data_filename_string.format(kwargs['itype'])
                else:
                    self.filename = CXData.processed_filename_string.format(kwargs['itype'])

                self.init_data(kwargs['itype'])

                self.__class__.__all__[kwargs['itype']]=self

            if 'data' in kwargs.keys():
                if isinstance(kwargs['data'], list):
                    self.data = kwargs['data']
                elif isinstance(kwargs['data'], np.ndarray):
                    self.data = [kwargs['data']]

    def __repr__(self):
        try:
            s=repr(self.data[0])
        except:
            s=''
        try:
            return '<{} at {}>\nA list of {} arrays with size {:d}x{:d}.\nShowing array 0 only:\n{}'.format(self.__class__,
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

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['log']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.log = CXP.log

    @staticmethod
    def inner_product(u, v):
        return u*v

    @staticmethod
    def proj_u_v(u, v):
        return (CXData.inner_product(u, v)/CXData.inner_product(u, u))*u

    def orthogonalise_states(self):
        cumulative_projections = None
        cdata = [element.copy() for element in self.data]
        for i in range(len(self.data)):
            cumulative_projections = np.zeros_like(self.data[0])
            if i==0:
                continue
            for j in range(i):
                cumulative_projections += CXData.proj_u_v(self.data[j], self.data[i])
            cdata[i] = self.data[1]-cumulative_projections
        self.data = cdata

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
            self.log.info('Rebasing data from 0 to {:3.2f}'.format(val))
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
            self.log.info('Taking square root.')
        for i in xrange(len(self.data)):
            self.data[i] = pow(self.data[i], 0.5)

    def fft_shift(self):
        if CXP.reconstruction.verbose:
            self.log.info('Performing FFT shift.')
        for i in xrange(len(self.data)):
            self.data[i] = spf.fftshift(self.data[i])

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

    def choose_actions(self):

        _CXP_path = '/'.join([self._CXP_dir, '_CXParams.py'])
        if CXP.actions.automate:
            print ast_buf
            self.log.info('Determining if data needs to be preprocessed.')
            print ast_buf
            for action in CXP.actions.__dict__:
                setattr(CXP.actions, action, False)

            if not os.path.isfile(_CXP_path):
                self.log.info('This appears to be the first time CXPhasing has been run.')
                self.log.info('Will run minimal reconstruction from raw data.')
                try:
                    shutil.copy('/'.join([CXP.io.code_dir, 'CXParams.py']), _CXP_path)
                except IOError:
                    raise Exception('Could not find CXParams.py. Update io.code_dir to point to the CXPhasing directory.')
                CXP.actions.preprocess_data = True
                CXP.actions.do_phase_retrieval = True
                return

            sys.path.append(self._CXP_dir)
            import _CXParams as _CXP
            diff = {}
            for key in CXP.__dict__.keys():
                element = getattr(CXP, key)
                if isinstance(element, CXP.param_store):
                    for entry in element.__dict__:
                        try:
                            if element[entry] != getattr(_CXP, key).__dict__[entry]:
                                diff[element] = key
                                self.log.info('Parameter change detected: {} in {}.'.format(entry, key))
                        except KeyError:
                            diff[element]=key
                            self.log.info('Parameter change detected: {} in {}.'.format(entry, key))

            for element in diff:
                if element not in ['reconstruction']:
                    setattr(CXP.actions, 'preprocess_data', True)
                    setattr(CXP.actions, 'do_phase_retrieval', True)
                else:
                    setattr(CXP.actions, 'do_phase_retrieval', True)

        for element in CXP.actions.__dict__:
            if getattr(CXP.actions, element) and element is not 'automate':
                self.log.info('Will perform {}'.format(element))

        if not os.path.exists(self._CXP_dir):
            self.log.info('Making new .CXPhasing directory')
            os.mkdir(self._CXP_dir)

        try:
            shutil.copy('/'.join([CXP.io.code_dir, 'CXParams.py']), _CXP_path)
        except IOError:
            self.log.error('Could not find CXParams.py. Update io.code_dir to point to the CXPhasing directory.')
            raise

    def setup_dir_tree(self):
        """Setup the directory structure for a new scan id"""

        if not os.path.exists(self._top_dir):
            self.log.info('Setting up new scan directory...')
            os.mkdir(CXData._top_dir)
            os.mkdir(CXData._sequence_dir)
            os.mkdir(CXData._cur_sequence_dir)
            os.mkdir(CXData._raw_data_dir)
            os.mkdir(CXData._dpc_dir)
            os.mkdir(CXData._CXP_dir)
            os.mkdir(CXData._py_dir)
            try:
                shutil.copy(CXP.io.code_dir+'/CXParams.py', CXData._py_dir)
            except IOError:
                self.log.error('Was unable to save a copy of CXParams.py to {}'.format(CXData._py_dir))
        else:
            self.log.info('Dir tree already exists.')
            if not os.path.exists(self._sequence_dir):
                os.mkdir(self._sequence_dir)
            if not os.path.exists(self._cur_sequence_dir):
                self.log.info('Making new sequence directory')
                os.mkdir(self._cur_sequence_dir)
            try:
                shutil.copy(CXP.io.code_dir+'/CXParams.py', CXData._py_dir)
                shutil.copy(CXP.io.code_dir+'/CXParams.py',
                            CXData._cur_sequence_dir+'/CXParams_sequence{}.py'.format(CXP.reconstruction.sequence))
            except IOError:
                self.log.error('Was unable to save a copy of CXParams.py to {}'.format(CXData._py_dir))

    def rot90(self, i):
        # Rotate by 90 degrees i times
        if CXP.reconstruction.verbose:
            self.log.info('Rotating data by {:d}'.format(i*90))
        for j, data in enumerate(self.data):
            self.data[j] = sp.rot90(data, i)

    def find_dead_pixels(self):
        # Return coordinates of pixels with a standard deviation of zero

        dead_pix = sp.where(abs(np.std(self.data, axis=0))<machine_precision)
        if CXP.reconstruction.verbose:
            self.log.info('Found {0:d} dead pixels'.format(len(dead_pix)))
        return dead_pix

    def zero_dead_pixels(self):
        if CXP.reconstruction.verbose:
            self.log.info('Setting dead pixels to zero')
        self.data[self.find_dead_pixels()]=0.

    def threshhold(self, threshhold=None):
        if not threshhold:
            threshhold = CXP.preprocessing.threshhold_raw_data
        if CXP.reconstruction.verbose:
            self.log.info('Applying threshhold to data at {:3.2f} and rebasing to 0.'.format(threshhold))

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
            self.log.info('Symmetrizing array shape.\n\tCurrent shape:\t{}x{}\n\tNew shape:\t{}x{}\n\tCentred on:\t{},{}'.format(
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
            filepath = self.filename
        try:
            self.log.info('Saving {} to:\n\t{}'.format(self.itype, filepath))
        except AttributeError:
            self.log.info('Saving to:\n\t{}'.format(filepath))
        try:
            np.savez(filepath, *self.data)
        except IOError as e:
            self.log.error(e)
            raise Exception('Could not save {} to {}'.format(self.kwargs['itype'], path))

    def load(self, path=None):
        if path:
            filepath = path
        else:
            filepath = self.filename
        self.log.info('Loading data from:\n\t{}'.format(filepath))
        try:
            self.data = self.openup(filepath)
        except IOError as e:
            self.log.error(e)
            raise Exception('Could not load file from {}'.format(filepath))
        if not isinstance(self.data, list):
            self.data = [self.data]

    def ptycho_mesh(self):
        """
        Generate a list of ptycho scan positions.

        Outputs
        -------
        self.data : list of 2xN arrays containing horizontal and vertical scan positions in pixels
        self.initial : initial guess at ptycho scan positions (before position correction)
        self.initial_skew : initial skew
        self.initial_rot : initial rotation
        self.initial_scl : initial scaling
        self.skew : current best guess at skew
        self.rot : current best guess at rotation
        self.scl : current best guess at scaling
        self.total : total number of ptycho positions

        [optional]
        self.correct : for simulated data this contains the correct position

        """
        self.log.info('Getting ptycho position mesh.')
        self.calc_ob_shape()
        self.calc_sam_pixel_size()

        if CXP.measurement.ptycho_scan_mesh == 'generate':
            if CXP.measurement.ptycho_scan_type == 'cartesian':
                x2 = 0.5*(CXP.measurement.cartesian_scan_dims[0]-1)
                y2 = 0.5*(CXP.measurement.cartesian_scan_dims[1]-1)
                tmp = map(lambda a: CXP.measurement.cartesian_step_size*a, np.mgrid[-x2:x2+1, -y2:y2+1])
                self.data = [tmp[0].flatten(), tmp[1].flatten()]
                if CXP.reconstruction.flip_mesh_lr:
                    self.log.info('Flip ptycho mesh left-right')
                    self.data[0] = self.data[0][::-1]
                if CXP.reconstruction.flip_mesh_ud:
                    self.log.info('Flip ptycho mesh up-down')
                    self.data[1] = self.data[1][::-1]
                if CXP.reconstruction.flip_fast_axis:
                    self.log.info('Flip ptycho mesh fast axis')
                    tmp0, tmp1 = self.data[0], self.data[1]
                    self.data[0], self.data[1] = tmp1, tmp0
            if CXP.measurement.ptycho_scan_type == 'round_roi':
                self.data = list(round_roi(CXP.measurement.round_roi_diameter, CXP.measurement.round_roi_step_size))
            if CXP.measurement.ptycho_scan_type == 'list':
                l = np.genfromtxt(CXP.measurement.list_scan_filename)
                x_pos, y_pos = [], []
                for element in l:
                    x_pos.append(element[0])
                    y_pos.append(element[1])
                self.data = [sp.array(x_pos), sp.array(y_pos)]


        elif CXP.measurement.ptycho_scan_mesh == 'supplied':
            l = np.genfromtxt(CXP.measurement.list_scan_filename)
            x_pos, y_pos = [], []
            for element in l:
                x_pos.append(element[0])
                y_pos.append(element[1])
            self.data = [sp.array(x_pos), sp.array(y_pos)]

        for element in self.data:
            element /= CXP.dx_s
            element += CXP.ob_p/2
        self.total = len(self.data[0])

        self.correct = [sp.zeros((self.total))]*2
        jit_pix = CXP.reconstruction.initial_position_jitter_radius
        search_pix = CXP.reconstruction.ppc_search_radius

        self.data[0] += jit_pix * uniform(-1, 1, self.total)
        self.data[1] += jit_pix * uniform(-1, 1, self.total)

        if CXP.reconstruction.probe_position_correction:
            self.correct[0] = self.data[0]+0.5*search_pix * uniform(-1, 1, self.total)
            self.correct[1] = self.data[1]+0.5*search_pix * uniform(-1, 1, self.total)
        else:
            self.correct = [self.data[0].copy(), self.data[1].copy()]

        data_copy = CXData(data=list(self.data))
        if not CXP.reconstruction.ptycho_subpixel_shift:
            self.data = [np.round(self.data[0]), np.round(self.data[1])]
            self.correct = [np.round(self.correct[0]), np.round(self.correct[1])]

        CXP.rms_rounding_error = [None]*2

        for i in range(2):
            CXP.rms_rounding_error[i] = sp.sqrt(sp.sum(abs(abs(data_copy.data[i])**2.-abs(self.data[i])**2.)))

        self.log.info('RMS Rounding Error (Per Position, X, Y):\t {:2.2f}, {:2.2f}'.format(CXP.rms_rounding_error[0]/len(self.data[0]),
                                                                                       CXP.rms_rounding_error[1]/len(self.data[1])))
        self.initial = [self.data[0].copy(), self.data[1].copy()]
        self.initial_skew = 0
        self.initial_rot = 0
        self.initial_scale = 0
        self.skew = 0
        self.rot = 0
        self.scale = [0, 0]

    def init_probe(self, *args, **kwargs):
        try:
            CXP.dx_s
        except AttributeError:
            self.calc_sam_pixel_size()

        dx_s = CXP.dx_s
        try:
            p, p2 = CXP.preprocessing.desired_array_shape, CXP.preprocessing.desired_array_shape/2
        except:
            p, p2 = CXP.experiment.px, CXP.experiment.px/2
        probe = sp.zeros((p, p), complex)

        if CXP.experiment.optic.lower() == 'kb':
            if len(CXP.experiment.beam_size)==1:
                bsx=bsy=np.round(CXP.experiment.beam_size[0]/dx_s)
            elif len(CXP.experiment.beam_size)==2:
                bsx, bsy = np.round(CXP.experiment.beam_size[0]/dx_s), np.round(CXP.experiment.beam_size[1]/dx_s)

            #probe+=machine_precision
            probe = np.sinc((np.arange(p)-p2)/bsx)[:,np.newaxis]*np.sinc((np.arange(p)-p2)/bsy)[np.newaxis,:]
            ph_func = vu.gauss_smooth(np.random.random(probe.shape), 10)
            probe=probe.astype(complex)
            probe =abs(probe)* exp(complex(0.,np.pi)*ph_func/ph_func.max())

        elif CXP.experiment.optic.lower() == 'zp':
            probe = sp.where(sp.hypot(*sp.ogrid[-p2:p2, -p2:p2])<np.round(CXP.experiment.beam_size[0]/(2*CXP.dx_s)), 1., 0.)
            ph = spn.gaussian_filter(uniform(-1, 1, p**2).reshape(probe.shape), sigma=3)
            #probe = spn.gaussian_filter(probe, sigma=0.5)
            probe = probe*exp(complex(0., np.pi)*ph)
        self.data = [probe]
        try:
            probe_det_mod = self.__class__.__all__['probe_det_mod']
            self.data[0] = spf.ifft2(probe_det_mod.data[0]*exp(complex(0., 1.)*sp.angle(spf.fft2(self.data[0]))))
        except (KeyError, AttributeError):
            pass

    def init_data(self, *args, **kwargs):

        if args[0]=='probe':
            self.init_probe()

        elif args[0] == 'object':
            self.calc_ob_shape()
            self.data = [sp.zeros((CXP.ob_p, CXP.ob_p), complex)]

        elif args[0] == 'psi':
            n_pos = len(self.__class__.__all__['positions'].data[0])
            p=CXP.preprocessing.desired_array_shape
            self.data = [sp.zeros((p, p), complex) for i in xrange(n_pos)]

        elif args[0] == 'positions':
            self.ptycho_mesh()

        elif args[0] == 'det_mod':
            if CXP.actions.preprocess_data:
                self.read_in_data()
            else:
                self.load()

        elif args[0] == 'probe_det_mod':
            if CXP.actions.preprocess_data:
                #  Get list of white files
                self.log.info('Preprocessing probe detector modulus.')
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
                self.log.info('Applied probe modulus constraint.')
            except (AttributeError, KeyError):
                pass

        elif args[0] == 'dark':
            if CXP.actions.preprocess_data:
                # Get list of dark files
                self.log.info('Preprocessing darkfield.')
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

        elif args[0] == 'probe_mod_squared':
            p = CXP.preprocessing.desired_array_shape
            self.data = [sp.zeros((p, p))]

        elif args[0] == 'probe_mod_squared_map':

            self.data = [sp.zeros((CXP.ob_p, CXP.ob_p))]

        elif args[0] == 'ob_mod_squared_map':

            self.data = [sp.zeros((CXP.ob_p, CXP.ob_p))]

    def read_in_data(self):

        self.completed_filenames = []  # Keep track of what's been processed already for online analysis
        self.job_filenames = []  # Bundle stack of images for preprocessing
        self.pattern = None

        # Determine which files to read in
        self.log.info('Reading in & preprocessing raw data...')

        #Pattern 1: 'image_{:d}.xxx'
        #Pattern 2: 'image_{:d}_{:d}.xxx'
        #Pattern 3: 'image_{:d}_{:d}_{val}.xxx'

        if self.pattern is None:  # Pattern is not yet dertermined

            filename, filerange, n_acqs = [CXP.io.data_filename, CXP.io.data_filename_range, CXP.measurement.n_acqs_data]

            self.pattern = filename.count('{')
            self.log.info('Detected filename pattern: {:d}'.format(self.pattern))
            if self.pattern == 0:
                raise Exception('NamingConventionError:\nPlease read CXParams for more info on file naming conventions.')

        try:
            n0, n1 = filerange[0], filerange[1]+1
        except IndexError:
            n0 = n1 = filerange[0]

        if CXP.io.darkfield_filename is not '':  # dark
            try:
                dark = self.__class__.__all__['dark']
                self.log.info('Found darkfield.')
            except KeyError:
                dark = CXData(itype='dark')
                dark.save()
        else:
            self.log.info('Not processing darkfields.')
            dark = None

        if CXP.io.whitefield_filename is not '':  # white
            try:
                probe_det_mod = self.__class__.__all__['probe_det_mod']
                self.log.info('Found probe detector modulus.')
            except KeyError:
                probe_det_mod = CXData(itype='probe_det_mod')
                probe_det_mod.save()
        else:
            self.log.info('Not processing whitefields.')
            probe_det_mod = None

        old_verbosity = CXP.reconstruction.verbose
        CXP.reconstruction.verbose = False

        jobs = mp.JoinableQueue()
        results = mp.Queue()

        if CXP.machine.n_processes<0:
            n_processes = mp.cpu_count()

        then = time.time()
        cnt=0
        missing_frames = False
        l=[]
        self.log.info('Dividing raw data into jobs over {:d} processes.'.format(n_processes))

        for i in range(n0, n1):
            if self.pattern == 1:
                s = [filename.format(i)]
            else:
                s = glob.glob((filename.split('}')[0]+'}*').format(i))
            # Include only files that haven't been processed yet
            # s = [fn for fn in s if fn not in self.completed_filenames]
            if len(s)==0:
                self.log.error('Globbed 0 files in CXData@read_in_files')
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
                    self.log.error('Missing frame: {:s}'.format(filename.format(i)))
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

        self.log.info('{:3.2f} seconds elapsed dividing jobs between processes.'.format(time.time()-then))
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

        self.log.info('{:3.2f} seconds elapsed preprocessing data.'.format(time.time()-then))
        CXP.reconstruction.verbose = old_verbosity

        self.save()

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

        # Save out processed data
        try:
            data.save()
            if white is not None:
                white.save()
        except AttributeError:
            pass

        return (stack_num, data.data)

    def calc_sam_pixel_size(self):
        try:
            p=CXP.preprocessing.desired_array_shape
        except:
            CXP.p=p
            self.log.warning('Using {:d} pixels for dx_s calculation'.format(CXP.p))

        # Calculate reconstruction plane pixel size
        CXP.dx_s = vu.energy_to_wavelength(CXP.experiment.energy)*CXP.experiment.z/(p*CXP.experiment.dx_d)
        self.log.info('Sample plane pixel size: {0:2.2e}'.format(CXP.dx_s))

    def calc_ob_shape(self):
        # Calculate reconstruction plane pixel size
        if CXP.measurement.ptycho_scan_type == 'cartesian':
            try:
                p=CXP.preprocessing.desired_array_shape
            except:
                CXP.p=p=CXP.experiment.px

            CXP.ob_p = np.int(2**(np.ceil(np.log2(p+((max(CXP.measurement.cartesian_scan_dims[0],
                                                              CXP.measurement.cartesian_scan_dims[1])-1)*
                                                              CXP.measurement.cartesian_step_size)))))
        elif CXP.measurement.ptycho_scan_type == 'round_roi':
            try:
                CXP.dx_s
            except:
                self.calc_sam_pixel_size()

            CXP.ob_p = np.int(pow(2., np.ceil(np.log2(CXP.measurement.round_roi_diameter/CXP.dx_s)+1)))
        if CXP.measurement.ptycho_scan_type == 'list':
            CXP.ob_p = 0
        CXP.ob_p = 512#max(1024, CXP.ob_p)
        self.log.info('Object shape: {0:d}'.format(CXP.ob_p))
        return CXP.ob_p

    def simulate_diffraction_patterns(self, sample):

        """
        This is a convenience function to generate simulated data for testing purposes.

        Values from CXParams are used to describe the experiment and measurement.

        Inputs
        ------
        sample : a CXData instance which will be use to generate the far field diffraction patterns.

        Outputs
        -------
        self.data : a stack of far-field diffraction patterns.

        """
        if not isinstance(sample, CXData):
            raise Exception('{} must be an instance of CXData'.format(sample))
        try:
            scan_positions = self.__class__.__all__['positions']
            probe = self.__class__.__all__['probe']
        except KeyError:
            self.calc_sam_pixel_size()
            scan_positions = CXData(itype='positions')
            probe = CXData(itype='probe')
            self.data = []

        p = CXP.preprocessing.desired_array_shape
        probe_large = CXData(data=sp.zeros((CXP.ob_p, CXP.ob_p), complex))
        probe_large[CXP.ob_p/2-p/2:CXP.ob_p/2+p/2, CXP.ob_p/2-p/2:CXP.ob_p/2+p/2] = probe
        p2 = p/2
        x, y = scan_positions.correct

        for i in xrange(len(x)):
            if i%(len(x)/10)==0.:
                self.log.info('Simulating diff patt {:d}'.format(i))
            tmp = abs(fft2(sample*probe_large[CXP.ob_p-x[i]-p2:CXP.ob_p-x[i]+p2, CXP.ob_p-y[i]-p2:CXP.ob_p-y[i]+p2]))
            self.data.append(tmp.data[0])

    def bin(self, n=None):

        """
        Bin a square array by grouping nxn pixels.
        Array size must be a multiple of n.

        """
        if n is None:
            n=CXP.preprocessing.bin
            # Now the detector pixel size has changed so we should update that
            CXP.experiment.dx_d *= n
            self.log.info('After binning new detector pixel size: {2.2e}'.format(CXP.experiment.dx_d))

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
            pylab.matshow(angle(self.data[i]))
        else:
            if log:
                pylab.matshow(sp.log(abs(self.data[i])))
            else:
                pylab.matshow(abs(self.data[i]))
        pylab.colorbar()
        pylab.show()

    def plot(self, i=0, phase=False):
        pylab.figure()
        if phase:
            pylab.plot(np.angle(self.data[i][:, CXP.ob_p/2]), label='Horizontal')
            pylab.plot(np.angle(self.data[i][CXP.ob_p/2, :]), label='Vertical')
        else:
            pylab.plot(np.abs(self.data[i][:, CXP.ob_p/2]), label='Horizontal')
            pylab.plot(np.abs(self.data[i][CXP.ob_p/2, :]), label='Vertical')
        pylab.legend()

    def calc_stxm_image(self):
        path = self._cur_sequence_dir+'/stxm_regular_grid.png'
        self.log.info('Calculating STXM image.\nSTXM saved to:\n\t{}'.format(path))
        image_sum = sp.array([sp.sum(data) for data in self.data])
        trigger_delete = False
        try:
            positions = self.__class__.__all__['positions']
        except (AttributeError, KeyError):
            positions = CXData(itype='positions')
            trigger_delete = True

        x, y = positions.data

        fig = Figure(figsize=(6, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title('STXM Image', fontsize=14)
        ax.set_xlabel('Position [micron]', fontsize=12)
        ax.set_ylabel('Position [micron]', fontsize=12)

        if CXP.measurement.ptycho_scan_type == 'cartesian':
            ax.hexbin(x, y, C=image_sum, gridsize=CXP.measurement.cartesian_scan_dims, cmap=cm.RdGy)
            canvas.print_figure(self._cur_sequence_dir+'/stxm_scatter', dpi=500)
            ax.imshow(image_sum.reshape(CXP.measurement.cartesian_scan_dims), cmap=cm.RdGy)
        else:
            ax.hexbin(x, y, C=image_sum, cmap=cm.RdGy)

        canvas.print_figure(path, dpi=500)

        if trigger_delete:
            del positions

    @staticmethod
    def _tukeywin(self, window_length, alpha=0.5):
        '''
        The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
        that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
        at \alpha = 0 it becomes a Hann window.

        We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
        output

        Reference
        ---------
        http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html

        '''
        # Special cases
        if alpha <= 0:
            return np.ones(window_length)  # rectangular window
        elif alpha >= 1:
            return np.hanning(window_length)

        # Normal case
        x = np.linspace(0, 1, window_length)
        w = np.ones(x.shape)

        # first condition 0 <= x < alpha/2
        first_condition = x<alpha/2
        w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2)))

        # second condition already taken care of

        # third condition 1 - alpha / 2 <= x <= 1
        third_condition = x>=(1 - alpha/2)
        w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

        return w[:, np.newaxis]*w[np.newaxis, :]

    def zstack_to_gif(self, log=False, filename=None):
        """
        makes a gif from the self.data

        Inputs:
        self.data - the current data stack

        Outputs:
        a gif of the data stack stored at self.filename

        """
        if hasattr(self, 'filename'):
            filename = self.filename
        elif filename:
            gif_filename = filename.split('.')[0]+'.gif'
        else:
            self.log.error('Must define a filename to create a gif')
            return
        print 'Writing GIF to {:s}'.format(gif_filename)
        l=[]
        for element in self.data:
            tmp = spf.fftshift(np.copy(element))
            if log:
                tmp = sp.log(tmp+0.01)
            tmp -= tmp.min()
            tmp /= tmp.max()
            l.append((255*tmp).astype('int'))
        writeGif(gif_filename, l, duration=0.2)

    def copy(self):
        return CXData(data=np.copy(self.data[0]))
