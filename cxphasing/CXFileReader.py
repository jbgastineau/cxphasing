import Image
import readMDA
import h5py
import os
import numpy
from mmpad_image import open_mmpad_tif
import numpy as np
import scipy as sp
import sys
#import libtiff

from cxparams import CXParams as CXP


class CXFileReader(object):

    """
    file_reader

    A generic and configurable file reader.

    The file reader determines the file type from the extension.

    For hierarchical data files a method for extracting the data must be specified.

    Inputs
    ------
    filename - the name of the file to read

    h5_file_path - hdf5 files: a string describing the location of the data inside a hierarchical data format
    mda_filepath - mda files: must specify whether to read a detector channel or positioner number.
                              For e.g. detector channel 5 mda_filepath='d5'
                                       positioner number 2 mda_filepath='p2'

    Outputs
    -------
    data - the 2 or 3D array read from the data file.

    Example Usage:

    fr = FileReader()
    data=fr.open('filename.h5', h5_file_path='/some/string')
    data=fr.open('filename.mda', mda_file_path='d4')  for detector channel 4


    """

    def __init__(self, *args, **kwargs):

        self.args = args

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def openup(self, filename, **kwargs):

        if not os.path.isfile(filename):
            CXP.log.error('{} is not a valid file'.format(filename))
            sys.exit(1)

        self.extension = filename.split('.')[-1].lower()
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        try:
            action = {
                        'mda': self.read_mda,
                        'h5':   self.read_h5,
                        'hdf5': self.read_h5,
                        'jpg': self.read_image,
                        'jpeg': self.read_image,
                        'png': self.read_image,
                        'tif': self.read_image,
                        'tiff': self.read_tif,
                        'npy':  self.read_npy,
                        'npz':  self.read_npz,
                        'dat':  self.read_dat,
                        'pkl':  self.read_pickle,
                        'mmpd': self.read_mmpad,
                        'pil': self.read_pilatus
                    }[self.extension]
        except NameError:
            CXP.log.error('Unknown file extension {}'.format(self.extension))
            raise

        return action(filename=filename)

    def read_mda(self, filename=None):
        if not filename:
            filename = self.filename

        source = self.mda_file_path[0].lower()
        if source not in ['d', 'p']:
            CXP.log.error("mda_file_path first character must be 'd' or 'p'")
            raise
        channel = self.mda_file_path[1]
        if not np.isnumeric(channel):
            CXP.log.error("mda_file_path second character must be numeric.")
            raise

        try:
            return readMDA.readMDA(filename)[2][source].data
        except:
            CXP.log.error('Could not extract array from mda file')
            raise

    def read_h5(self, filename=None, h5_file_path='/entry/instrument/detector/data'):

        if not filename:
            filename = self.filename

        try:
            h5_file_path = self.h5_file_path
        except:
            pass

        try:
            return h5py.File(filename)[h5_file_path].value
        except:
            CXP.log.error('Could not extract data from h5 file.')
            raise

    def read_image(self, filename=None):
        if not filename:
            filename = self.filename
        try:
            return sp.misc.fromimage(Image.open(filename))
        except:
            CXP.log.error('Unable to read data from {}'.format(filename))
            raise

    def read_npy(self, filename=None):

        if not filename:
            filename = self.filename
        try:
            return numpy.load(filename)
        except IOError as e:
            print e
            CXP.log.error('Could not extract data from numpy file.')
            raise

    def read_npz(self, filename=None):
        if not filename:
            filename = self.filename
        l=[]
        try:
            d= dict(numpy.load(filename))
            # Return list in the right order
            for i in range(len(d)):
                l.append(d['arr_{:d}'.format(i)])
            return l
        except IOError:
            CXP.log.error('Could not extract data from numpy file.')
            raise

    def read_dat(self, filename=None):
        if not filename:
            filename = self.filename
        try:
            return sp.fromfile(filename)
        except:
            CXP.log.error('Could not extract data from data file.')
            raise

    def read_pickle(self, filename=None):
        if not filename:
            filename = self.filename

        try:
            return pickle.load(filename)
        except:
            CXP.log.error('Could not load data from pickle')
            raise

    def read_mmpad(self, filename=None):
        if not filename:
            filename = self.filename

        try:
            return open_mmpad_tif(filename)
        except:
            CXP.log.error('Could not load data from pickle')
            raise

    def read_pilatus(self, filename=None):
        if not filename:
            filename = self.filename
        try:
            return sp.misc.fromimage(Image.open(filename))[:-1,:-1]
        except:
            CXP.log.error('Unable to read data from {}'.format(filename))
            raise

    def read_tif(self, filename=None):
        if not filename:
            filename = self.filename
        try:
            return libtiff.TIFF.open(filename).read_image()
        except:
            CXP.log.error('Unable to read data from {}'.format(filename))
            raise
