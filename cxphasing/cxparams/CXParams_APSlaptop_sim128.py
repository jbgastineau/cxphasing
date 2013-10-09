"""

CXParams.py

CXParams stores all of the parameters for analysing a set of data.

It is the only interface to the phase retrieval process that the user needs to interact with.

"""


import os
import pdb
import numpy as np
import logging
import logging.handlers
import cxphasing.CXUtils as CXU

class param_store(object):
  """
  param_store is a wrapper class for storing parameters.
  """
  instances = []
  def __init__(self, name):
    param_store.instances.append(name)
  def __getitem__(self, s):
    return self.__dict__[s]

version =         0.5
db =              param_store('db')
machine =         param_store('machine')
actions =         param_store('actions')
io =              param_store('io')
experiment =      param_store('experiment') 
measurement =     param_store('measurement')
preprocessing =   param_store('preprocessing')
reconstruction =  param_store('reconstruction')

"""
--------
Database
--------

A MySQL database is used to record all the details of each reconstruction attempt.

All parameters are stored at the bginning of the run and various derived quantities are stored
as they are calculated.

"""

db.dbhost = 'localhost'
db.dbuser = 'david'
db.dbpass = 'david'
db.master_db = 'CXP_master'


"""
-------
Machine 
-------

Parameters related to the machine on which the reconstruction is running.

n_processes: define how many processes to use for parallel computation. 
             Default: -1 uses the number of cores on machine.

"""

machine.n_processes = -1

"""
-------
Actions
-------

Parameters related to the overview of what should happen during the reconstruction.

"""

actions.automate =              False
actions.preprocess_data =       False
actions.process_dpc =           False
actions.do_phase_retrieval =    True


"""
--
IO
--

Parameters describing interaction with the filesystem.

Inputs
------
scan_id: a unique string used to create an outout folder for reconstructions

base_dir: an optinal string that will be prepended to identify data locations

code_dir: a copy of the source code is zipped and saved with each reconstruction.

log_dir: specifies the logfile directory

mda_file: location of mda_file

data_filename: a string describing the pattern to identify data frames.
               The pattern specifies how the data should merged and includes 
               repeated acquisitions and multiple exposure times.
               Single exposure time:
                  Two patterns can be used:
                    (a) data_filename = 'image_{0:06d}.h5'
                    (b) data_filename = 'image_{0:06d}_{0:05d}.h5'
                  In (b) it is assumed that the final number field represents
                  repeated acquisitions.
                Multiple exposure times and acqisitions (per position):
                  Only one pattern can be used:
                  (c) data_filename = 'image_{0:06d}_{0:05d}_{exp time in msec}.h5'

data_filename_range: a list with an entry for each variable defined in data_filename.
                        e.g data_filename_range = [0,199] means image_00000.h5 to image_00199.h5
                      Similarly for whitefield and darkfield


"""

io.scan_id =                    'sim4'
io.base_dir =                   '/home/david/data/simulation'
io.code_dir =                   '/home/david/python/CXPhasing/cxphasing'
io.log_dir  =                   '/var/log/CXPhasing'
io.mda_file =                   ''
io.data_filename =              ''
io.data_filename_range =        ''
io.whitefield_filename =        ''
io.whitefield_filename_range =  ''
io.darkfield_filename =         ''
io.darkfield_filename_range =   ''
io.simulation_sample_filename = ['/home/david/python/lena128.npy',
                                 '/home/david/python/einstein128.npy']


"""
----------
Experiment
----------

Parameters related to the experiment setup. These parameters are constant for a given experiment or beamline.
Parameters which change from measurement to measurement are in Measurement.

beam_size: 1- (or 2-) element list describing the beam size in the sample plane for zp/pinhole (or kb) focusing optics.

"""

experiment.beam_size =    [100e-9 ]
experiment.energy =        5.0 # keV
experiment.dx_d =          172e-6
experiment.px =            128
experiment.py =            128
experiment.z =             1.0
experiment.optic =         ['ZP', 'KB'][0]

"""
Measurement

Time taken for reconstruction depends critically on whether to perform subpixel shifting.
If the ptycho scan mesh positions are integers then no subpixel shifting is performed. If
they are not integers the algorithm will use an FFT based shifting function which pads the
array size to the next power of 2. This is computationally expensive to do and dramatically
slows down the time to analyse the data.

ptycho_subpixel_shift: describes whether to allow subpixel shifting. In practice, this
                       has the effect of rounding the scan_mesh values to the nearest 
                       integer.

"""
measurement.simulate_data =          True
measurement.n_acqs_data =            1
measurement.n_acqs_whitefield =      1
measurement.n_acqs_darkfield =       0
measurement.ptycho_scan_type =       ['cartesian', 'round_roi', 'list'][0]
measurement.ptycho_scan_mesh =       ['generate', 'supplied'][0]
measurement.cartesian_scan_dims =    (12, 12) # Used to generate Cartesian scan mesh
measurement.cartesian_step_size =    50e-9   # Used to generate Cartesian scan mesh 
measurement.round_roi_diameter =     2.5e-6    # Used to generate Round ROI scan mesh
measurement.round_roi_step_size =    250e-9   # Used to generate Round Roi scan mesh
measurement.list_scan_filename =     ''      # Used to generate scan mesh

"""
Preprocessing

This section defines how raw data is turned into processed data.

shift_data - a shift applied to all images in a stack
threshhold_raw_data - threshholds and rebases all images in stack


Choosing an ROI/Padding & Cropping data:

Changing the array size can be done by entering the pixel values corresponding to
qx=0 & qy=0 and then specifying the desired array size. The array will then be selected
symmetrically around the qx, qy position. If padding is necessary the padded values
will be zero.

qx0qy0 -             the xy coords (pixels) that define the qx=0, qy=0 in the detector
                     plane.
desired_array_size - desired array size in pixels. Setting to None will result in 
                     no changes made to the array size. The array will be cropped
                     to a square and preferably even sized.
bin -                 specify the binning factor

saturation_level -   detector pixels >= this value are assumed to be saturated when merging mutliple-
                     exposure-time acquisitions into a single diffraction pattern. Saturated pixels 
                     are set to zero as if they weren't measured.
                     Value below zero will skip any saturartion correction.

"""


preprocessing.shift_data =               (0,0)
preprocessing.threshhold_raw_data =      -1
preprocessing.saturation_level =         -1
preprocessing.qx0qy0 =                   (64, 64)
preprocessing.desired_array_shape =      128
preprocessing.object_array_shape =       1024
preprocessing.detector_type =            ['ccd', 'pad'][1]
preprocessing.rot90 =                    1
preprocessing.bin =                      1
preprocessing.calc_stxm_image =          True



"""
Reconstruction 
"""

reconstruction.sequence =                               4
reconstruction.ptycho_its =                             1000
reconstruction.begin_updating_probe =                   10
reconstruction.begin_modal_reconstruction =             20
reconstruction.algorithm =                              {'er':5, 'dm':2}
reconstruction.probe_object_its =                       1 # Number of times to update probe and object contiguouslyo'
reconstruction.verbose =                                True
reconstruction.flip_mesh_lr =                           False # Reverse ptycho grid left-right
reconstruction.flip_mesh_ud =                           False # Reverse ptycho grid up-down
reconstruction.flip_fast_axis =                         False # Swap fast axis from vertical to horizontal
reconstruction.ptycho_subpixel_shift =                  False
reconstruction.probe_position_correction =              False
reconstruction.ppc_begin =                              300 # Begin updating after N iterations
reconstruction.ppc_length =                             100 # Update positions for N iterations
reconstruction.ppc_trial_positions =                    4
reconstruction.ppc_search_radius =                      5 # pixels
reconstruction.initial_position_jitter_radius =         5
reconstruction.probe_modes =                            2





# Setup logger

if not os.path.exists(io.log_dir):
  os.mkdir(io.log_dir)
  open(io.log_dir+'/CXP.log', 'w')
  close(io.log_dir+'/CXP.log')
  os.chown(io.log_dir+'/CXP.log', os.getlogin())
log = logging.getLogger('CXPhasing')
log.setLevel(logging.DEBUG)
# Create a file handler for debug level and above
fh = logging.handlers.RotatingFileHandler(io.log_dir+'/CXP.log', maxBytes = 1e6, backupCount=5)
fh.setLevel(logging.DEBUG)
# Create console handler with higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
ch.setFormatter(console_formatter)
fh.setFormatter(file_formatter)
log.addHandler(ch)
log.addHandler(fh)

# Derived quanitites
p = preprocessing.desired_array_shape
ob_p = preprocessing.object_array_shape
dx_s = CXU.energy_to_wavelength(experiment.energy)*experiment.z/(p*experiment.dx_d)