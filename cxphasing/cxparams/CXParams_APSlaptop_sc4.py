"""

CXParams.py

CXParams stores all of the parameters for analysing a set of data.

It is the only interface to the phase retrieval process that the user needs to interact with.

"""


import os
import numpy as np

class param_store(object):
  """
  param_store is a wrapper class for storing parameters.

  _priority is used to automate the reconstruction process. The new parameter file is compared
  to the archived version to determine what has changed. If only parameters relating to the
  reconstruction changed the raw data does not need to be re-processed. 
  """

  _priority = []
  def __init__(self, name):
    param_store._priority.append(name)
  def __getitem__(self, s):
    return self.__dict__[s]

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
             Default: None uses the number of cores on machine.

"""

machine.n_processes = None

"""
-------
Actions
-------

Parameters related to the overview of what should happen during the reconstruction.

"""

actions.automate =              False
actions.preprocess_data =       True
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
                Two patterns can be used:
                  (a) data_filename = 'image_{0:06d}.h5'
                  (b) data_filename = 'image_{0:06d}_{0:05d}.h5'
                In (b) it is assumed that the final number field represents
                repeated acquisitions.


data_filename_range: a list with an entry for each variable defined in data_filename.
                        e.g data_filename_range = [0,199] means image_00000.h5 to image_00199.h5
                      Similarly for whitefield and darkfield


"""

io.scan_id =                    'sc4'
io.base_dir =                   '/home/david/data'
io.code_dir =                   '/home/david/python/CXPhasing'
io.log_dir  =                   '/var/log/CXPhasing/log'
io.mda_file =                   None
io.data_filename =              '/home/david/data/sxdm/sc4/mmpd/sc4_a_{:06d}_{:05d}.mmpd'
io.data_filename_range =        [0, 21**2]
io.whitefield_filename =        '/home/david/data/sxdm/sc6/tif/sc6_a_{:06d}_{:05d}.mmpd'
io.whitefield_filename_range =  [1, 2]
io.darkfield_filename =         None
io.darkfield_filename_range =   None


"""
----------
Experiment
----------

Parameters related to the experiment setup. These parameters are constant for a given experiment or beamline.
Parameters which change from measurement to measurement are in Measurement.

beam_size: 1- (or 2-) element list describing the beam size in the sample plane for zp/pinhole (or kb) focusing optics.

"""

experiment.beam_size =    [150e-9 ]
experiment.energy =        2.535 # keV
experiment.dx_d =          150e-6
experiment.px =            266
experiment.py =            396
experiment.z =             0.45
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

measurement.n_acqs_data =            10
measurement.n_acqs_whitefield =      40
measurement.n_acqs_darkfield =       None
measurement.ptycho_scan_type =       ['cartesian', 'round_roi', 'list'][0]
measurement.ptycho_scan_mesh =       ['generate', 'supplied'][0]
measurement.ptycho_subpixel_shift =  False
measurement.cartesian_scan_dims =    (21,21) # Used to generate Cartesian scan mesh
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

"""


preprocessing.shift_data =               () # Shift all data by x, y pix.
preprocessing.threshhold_raw_data =      None
preprocessing.qx0qy0 =                   (110,185)
preprocessing.desired_array_shape =      128
preprocessing.detector_type =            ['ccd', 'pad'][1]
preprocessing.rot90 =                    0
preprocessing.bin =                      None
preprocessing.calc_stxm_image =          True



"""
Reconstruction 
"""

reconstruction.sequence =                           1
reconstruction.auto_increment_sequence =            False
reconstruction.ptycho_its =                         300
reconstruction.ER_n =                               200 # Perform ER every nth iteration
reconstruction.begin_updating_probe =               30
reconstruction.probe_object_its =                   2 # Number of times to update probe and object contiguously
reconstruction.averaging_start =                    150 # Begin averaging at iteration 150
reconstruction.averaging_n =                        5 # Average every nth iteration
reconstruction.probe_regularisation =               'Auto'
reconstruction.object_regularisation =              'Auto'
reconstruction.probe_position_correction =          'Not implemented yet, sorry'
reconstruction.probe_position_correction_begin =    30
reconstruction.probe_position_correction_update =   5 
reconstruction.verbose =                            True  