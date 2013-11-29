"""
.. module:: CXPhasing.py
   :platform: Unix
   :synopsis: Implements phase retrieval algorithms.

.. moduleauthor:: David Vine <djvine@gmail.com>


""" 

import os
import numpy as np
import scipy as sp
from scipy.stats import poisson
import pylab
import time
import math
import pdb
from numpy.random import uniform, choice, normal
import multiprocessing as mp
import itertools
import shutil

import cxphasing.cxparams.CXParams as CXP
from CXData import fft2, ifft2, angle, exp, fftshift, conj, abs, sqrt
from CXData import log as nlog
from CXData import CXData, CXModal
from CXDb import SimpleDB
from CXUtils import worker, object_worker, split_seq, v_hls_to_rgb, energy_to_wavelength, gauss_smooth
import multiprocess

from matplotlib import cm
from matplotlib import rc
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colorbar import Colorbar

try:
    import MySQLdb
    hasmysql = True
except:
    hasmysql = False

hasmysql*=CXP.db.usemysql

class CXPhasing(object):
    """
    .. class:: CXPhasing(object)
        Implements phase retrieval process.


        :attr annealing_schedule: Annealing schedule for probe position correction
        :type annealing_schedule: lambda function
        :attr int p: side length of state vector array in pixels
        :attr int p2: half side length of state vector array in pixels
        :attr int ob_p: side length of object array in pixels
        :attr int total_its: the total number of iterations
        :attr int probe_modes: the number of probe modes
        :attr dict algorithms: dictionary of functions implementing iterative phase retrieval algorithms
        :attr algorithm: the current phase retrieval algorithm
        :type algorithm: lambda function
        :attr str em_repr: the update string for Error Reduction iterations
        :attr str dm_repr: the update string for Difference Map iterations
        :attr str progress_repr: the update string printed once per iteration
        :attr log: used for creating a log file and printing data to the terminal
        :type log: Logging object
        :attr int itnum: the current global iteration number
        :attr bool ppc: probe position correction


    """

    def __init__(self):
        # Annealing schedule for probe position correction
        self.annealing_schedule = lambda x: 1 if x ==0 else np.max([0.05,
                                    1. - np.double(x) / CXP.reconstruction.ppc_length])

        # Flag for performing position correction
        self.ppc = False

        # MySQL DB Integration
        self.db_queue = {}
        if hasmysql:
            self.init_db_conn()        

        self.p = CXP.p
        self.p2 = self.p / 2
        self.ob_p = CXP.preprocessing.object_array_shape
        self.total_its = 0
        self.probe_modes = CXP.reconstruction.probe_modes

        self.mask = CXData(data=sp.ones((self.p, self.p)))
        self.mask.data[0][0,:] = 0.
        self.mask.data[0][-1,:] = 0.
        self.mask.data[0][:,0] = 0.
        self.mask.data[0][:,-1] = 0.

        self.algorithm = 'er' # Start with error reduction

        if CXP.machine.n_processes < 0:
            CXP.machine.n_processes = mp.cpu_count()

        self.epie_repr = '{:s}\n\tPtychography iteration:{:10d}\n\tPtychography position:{:10d} [{:3.0f}%]'
        self.progress_repr = 'Current iteration: {:d}\tPosition: {:d}'
        
        self._sequence_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'sequences'])
        self._cur_sequence_dir = self._sequence_dir+'/sequence_{:d}'.format(CXP.reconstruction.sequence)
        self._raw_data_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'raw_data'])

    def setup(self):
        """
        .. method:: setup()

            This function implements all of the setup required to begin a phasing attempt.
             - Setup directory structure.
             - Initiliase the init_figure.
             - Log all slow parameters to the db.

            :param path: The path to the new CXParams file.
            :type path: str.
            :returns:  int -- the return code.
            :raises: IOError

        """
        self.setup_dir_tree()

        self.init_figure()

        self.log_reconstruction_parameters()

    def preprocessing(self):
        """.. method:: preprocessing()
            Collects together all the preprocessing functions that are required to begin phase retrieval.

        """
        # Get the scan positions
        self.positions = CXData(name='positions', data=[])
        self.ptycho_mesh()

        if CXP.measurement.simulate_data:
            self.simulate_data()
        else:
            # Read in raw data
            self.det_mod = CXData(name = 'det_mod')
            if CXP.actions.preprocess_data:
                self.det_mod.read_in_data()
                self.det_mod.save(path=self._raw_data_dir+'/det_mod.npz')
            else:
                self.det_mod.load(path=self._raw_data_dir+'/det_mod.npz')
            if CXP.io.whitefield_filename:
                self.probe_det_mod = CXData(name='probe_det_mod')
                self.probe_det_mod.preprocess_data()

        self.object = CXData(name='object', data=[sp.zeros((self.ob_p, self.ob_p), complex)])
        
        self.probe_intensity = CXData(name='probe_intensity', data=[sp.zeros((self.p, self.p))])

        self.probe = CXModal(modes=[])
    
        self.init_probe()

        # Calculate STXM image if this is a ptycho scan
        if len(self.det_mod.data) > 1:
            self.calc_stxm_image()

        if CXP.actions.process_dpc:
            self.process_dpc()


    def phase_retrieval(self):
        """.. method:: phase_retrieval()
            Runs the itertaive phase retrieval process.

        """ 
        its = CXP.reconstruction.ptycho_its

        beginning = time.time()
        
        for self.itnum in xrange(its):
            then = time.time()

            self.select_algorithm()

            self.ePIE()

            now = time.time()
            if hasmysql:
                self.db_queue['iter_time'] = (self.itnum, now - then)
                self.db_queue['iter_time_pptpxit'] = (self.itnum, 1e6*(now - then) / (self.positions.total * self.p**2 * (self.itnum + 1)))
            CXP.log.info('{:2.2f} seconds elapsed during iteration {:d} [{:1.2e} sec/pt/pix/it]'.format(now - then, self.itnum + 1,
                            (now-then)/(self.positions.total * self.p**2 * (self.itnum + 1))))
            CXP.log.info('{:5.2f} seconds have elapsed in {:d} iterations [{:2.2f} sec/it]'.format(now-beginning, self.itnum + 1, (now-beginning)/(self.total_its + 1)))
            if CXP.reconstruction.calc_chi_squared:
                self.calc_mse()
            self.total_its += 1
            if hasmysql:
                self.update_recon_param_table()
                #if self.total_its%10:
                    # Plot from db every 10th iteration
                #    self.db_plot()
            if self.itnum > 0:
                self.update_figure(self.itnum)

    def postprocessing(self):
        """.. method::postprocessing()
            Collectes together all the orutines that should be completed after the iterative phase retrieval has successfully completed.

        """
        pass

    def simulate_data(self, no_save=False):
        CXP.log.info('Simulating diffraction patterns.')
        self.sample = CXData()
        self.sample.load(CXP.io.simulation_sample_filename[0])
        self.sample.data[0] = self.sample.data[0].astype(float)
        self.sample.normalise(val=0.8)
        delta_T = CXP.simulation.sample_transmission[1]-CXP.simulation.sample_transmission[0]
        self.sample.data[0]+=CXP.simulation.sample_transmission[0]
        self.input_probe = CXModal()
        if len(CXP.io.simulation_sample_filename)>1:
            ph = CXData()
            ph.load(CXP.io.simulation_sample_filename[1])
            ph.data[0] = ph.data[0].astype(float)
            ph.normalise(val=CXP.simulation.sample_phase_shift)
            self.sample.data[0] = self.sample.data[0]*exp(complex(0., 1.)*ph.data[0])
        p = self.sample.data[0].shape[0]
        ham_window = sp.hamming(p)[:,np.newaxis]*sp.hamming(p)[np.newaxis,:]
        sample_large = CXData(data=sp.zeros((CXP.ob_p, CXP.ob_p), complex))
        sample_large.data[0][CXP.ob_p/2-p/2:CXP.ob_p/2+p/2, CXP.ob_p/2-p/2:CXP.ob_p/2+p/2] = self.sample.data[0]*ham_window

        ker = sp.arange(0, p)
        fwhm = p/3.0
        radker = sp.hypot(*sp.ogrid[-p/2:p/2,-p/2:p/2])
        gaussian = exp(-1.0*(fwhm/2.35)**-2. * radker**2.0 )
        ortho_modes = lambda n1, n2 : gaussian*np.sin(n1*math.pi*ker/p)[:,np.newaxis]*np.sin(n2*math.pi*ker/p)[np.newaxis, :]
        mode_generator = lambda : sp.floor(4*sp.random.random(2))+1

        used_modes = []
        self.input_psi = CXModal()
        
        for mode in range(CXP.reconstruction.probe_modes):
            if mode==0:
                new_mode = [1,1]
                mode_amp = 1.0
            else:
                new_mode = list(mode_generator())
                while new_mode in used_modes:
                    new_mode = list(mode_generator())
                mode_amp = (10*mode)**-1.
            used_modes.append(new_mode)
            CXP.log.info('Simulating mode {:d}: [{:d}, {:d}]'.format(mode, int(new_mode[0]), int(new_mode[1])))
            ph_func = gauss_smooth(np.random.random((p,p)), 10)
            self.input_probe.modes.append(CXData(name='probe{:d}'.format(mode), 
                data=mode_amp*ortho_modes(new_mode[0], new_mode[1])*exp(complex(0.,np.pi)*ph_func/ph_func.max())))
        
        self.input_probe.normalise()
        self.input_probe.orthogonalise()

        for mode in range(CXP.reconstruction.probe_modes):
            p2 = p/2
            x, y = self.positions.correct
            self.input_psi.modes.append(CXData(name='input_psi_mode{:d}'.format(mode), data=[]))
            
            for i in xrange(len(x)):
                if i%(len(x)/10)==0.:
                    CXP.log.info('Simulating diff patt {:d}'.format(i))
                tmp = (CXData.shift(sample_large, -1.0*(x[i]-CXP.ob_p/2), -1.0*(y[i]-CXP.ob_p/2))
                        [CXP.ob_p/2-p2:CXP.ob_p/2+p2, CXP.ob_p/2-p2:CXP.ob_p/2+p2]*
                        self.input_probe[mode][0])
                self.input_psi[mode].data.append(tmp.data[0])

        # Add modes incoherently
        self.det_mod = CXModal.modal_sum(abs(fft2(self.input_psi)))
        # Add noise
        if CXP.simulation.noise_model == 'poisson':
            for i in range(self.positions.total):
                self.det_mod.data[i] = poisson.rvs(self.det_mod.data[i])
        elif CXP.simulation.noise_model == 'gaussian':
            for i in range(self.positions.total):
                self.det_mod.data[i]+=self.det_mod.data[i]*CXP.simulation.gaussian_noise_level*normal(size=(CXP.p, CXP.p))
        

        # Limit total counts
        if CXP.simulation.total_photons>0:
            for i in range(self.positions.total):
                self.det_mod.data[i] -= self.det_mod.data[i].min()
                self.det_mod.data[i] *= sp.floor(CXP.simulate_data.total_photons/sp.sum(self.det_mod.data[i]))
                deficit = CXP.simulation.total_photons-sp.sum(self.det_mod.data[i])

        if not no_save:
            self.det_mod.save(path=CXP.io.base_dir+'/'+CXP.io.scan_id+'/raw_data/{:s}.npy'.format('det_mod'))

    def pos_correction_transform(self, i):
        # Generates trial position
        search_rad = CXP.reconstruction.ppc_search_radius

        itnum = self.itnum-CXP.reconstruction.begin_probe_position_correction

        r = self.annealing_schedule(itnum)

        cx = self.positions.data[0][i] + (search_rad * r * uniform(-1, 1))
        cy = self.positions.data[1][i] + (search_rad * r * uniform(-1, 1))

        # Limit max deviation
        if np.abs(cx - self.positions.initial[0][i]) > search_rad:
            cx = self.positions.initial[0][i] + 0.5*search_rad * r * uniform(-1, 1)
        if np.abs(cy - self.positions.initial[1][i]) > search_rad:
            cy = self.positions.initial[1][i] + 0.5*search_rad * r * uniform(-1, 1)

        if CXP.reconstruction.ptycho_subpixel_shift:
            return [cx, cy]
        else:
            return [np.round(cx), np.round(cy)]

    @staticmethod
    def relaxedM(psi, det_mod):
        """.. method:: M(mode, psi_modes, det_mod)

            Applies modulus constraint to psi_modes(mode) for a given position.

            :param list psi_modes: A list of CXData instances containing all modes at a given position.
            :param np.ndarray det_mod: Modulus of measured diffraction pattern.

        """
        threshhold = 0.625
        if isinstance(psi, CXData):
            psi_bar = fft2(psi)
            distance = abs(abs(psi_bar)-det_mod)
            modulus = CXData(data=sp.where(distance.data[0] <= threshhold, abs(psi_bar).data[0], (1.0-(threshhold/distance.data[0]))*det_mod.data[0] +
             (threshhold/distance.data[0])*abs(psi_bar.data[0])))
            return ifft2(modulus * exp(complex(0., 1.) * angle(psi_bar)))
        elif isinstance(psi, CXModal):
            mode_sum = CXModal.modal_sum(abs(fft2(psi))**2.0)**0.5
            distance = abs(mode_sum - det_mod)
            modulus = CXData(data=sp.where(distance.data[0] <= threshhold, mode_sum.data[0], (1.0-(threshhold/distance.data[0]))*det_mod.data[0] + 
                (threshhold/distance.data[0])*mode_sum.data[0]))
            return ifft2((fft2(psi)/(mode_sum))*modulus)

    @staticmethod
    def M(psi, det_mod):
        """.. method:: M(mode, psi_modes, det_mod)

            Applies modulus constraint to psi_modes(mode) for a given position.

            :param list psi_modes: A list of CXData instances containing all modes at a given position.
            :param np.ndarray det_mod: Modulus of measured diffraction pattern.

        """
        if isinstance(psi, CXData):
            return ifft2(det_mod * exp(complex(0., 1.) * angle(fft2(psi))))
        elif isinstance(psi, CXModal):
            if len(psi)==1:
                return ifft2(det_mod * exp(complex(0., 1.) * angle(fft2(psi[0]))))
            else:
                mode_sum = CXModal.modal_sum(abs(fft2(psi))**2.0)**0.5
                return ifft2((fft2(psi)/(mode_sum))*det_mod)

    def ePIE(self):
        """.. method:: ePIE(self)

            This method uses ePie to generate the initial estimate for psi and object.

        """
        
        d1, d2 = self.positions.data
        positions = range(self.positions.total)
        self.av_chisq = 0

        for it in xrange(self.positions.total):

            # Randomise analysis positions
            i = choice(positions)
            positions.remove(i)

            if it % np.floor(self.positions.total / 10) == 0 and CXP.reconstruction.verbose:
                CXP.log.info(self.epie_repr.format(self.algorithm_name, self.itnum, it, 100. * float(it + 1) / self.positions.total))
             
            view = self.probe * self.object[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]
            view_old = view.copy()
            
            if self.ppc:
                
                error0 = (sp.sum((abs(CXModal.modal_sum(abs(fft2(view))) - self.det_mod[i]) ** 2.).data[0]) / sp.sum(self.det_mod.data[i] ** 2.))**0.5
                
                ppc_dict = {}
                
                multip = multiprocess.multiprocess(self.epie_worker)
                
                for trial in range(CXP.reconstruction.ppc_trial_positions):
                    cx, cy = self.pos_correction_transform(i)
                    ppc_dict[trial] = (cx, cy)
                    multip.add_job((trial, self.probe * self.object[cx - self.p2:cx + self.p2, cy - self.p2:cy + self.p2], self.det_mod[i]))
                
                results = multip.close_out()
                
                for result in results:
                    trial, error = result
                    if error < error0:
                        error0 = error
                        self.positions.data[0][i], self.positions.data[1][i] = ppc_dict[trial]
                        view = self.probe * self.object[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]
                        view_old = view.copy()

            if self.algorithm == 'er':
                view= self.M(view, self.det_mod[i])
            elif self.algorithm == 'dm':
                view += self.M(2*view-self.M(view, self.det_mod[i]), self.det_mod[i]) - view
                
            self.update_object(i, view_old, view)
            if self.do_update_probe:
                self.update_probe(i, view_old, view)
        
        if self.do_update_probe:
            self.probe.normalise()

        for mode, probe in enumerate(self.probe.modes):
            probe.save(path=self._cur_sequence_dir+'/probe_mode{:d}.npy'.format(mode))
        self.object.save(path=self._cur_sequence_dir+'/object.npy')

        self.av_chisq /= self.positions.total
        self.db_queue['chisq'] = (self.itnum, self.av_chisq)

    @staticmethod
    @multiprocess.worker
    def epie_worker(args):
        trial, view, det_mod = args

        error =  (sp.sum((abs(CXModal.modal_sum(abs(fft2(view))) - det_mod) ** 2.).data[0]) / sp.sum(det_mod.data[0] ** 2.))**0.5
        
        return (trial, error)

    def update_object(self, i, psi_old, psi_new):
        """
        Update the object from a single ptycho position.

        """
        d1, d2 = self.positions.data
        id1, id2 = d1//1, d2//1
        probe_intensity_max = CXModal.modal_sum(abs(self.probe)**2.0).data[0].max()
        
        self.object[id1[i] - self.p2:id1[i] + self.p2, id2[i] - self.p2:id2[i] + self.p2] += \
            CXData.shift(CXModal.modal_sum(conj(self.probe) * (psi_new-psi_old)) / probe_intensity_max, 
                d1[i]%1, d2[i]%1)

        if self.total_its==0 and sp.mod(i, len(self.positions.data[0]) / 10) == 0:
            self.update_figure(i)

    def update_probe(self, i, psi_old, psi_new):
        
        d1, d2 = self.positions.data
        id1, id2 = d1//1, d2//1
        object_intensity_max = (abs(self.object)**2.0).data[0].max()
         
        for mode in range(len(self.probe)):
            self.probe.modes[mode] += \
                CXData.shift(conj(self.object[id1[i] - self.p2:id1[i] + self.p2, id2[i] - self.p2:id2[i] + self.p2]) *
                 (psi_new-psi_old)[mode] / object_intensity_max, d1[i]%1, d2[i]%1)

        self.probe.orthogonalise()

    def error(self, psi, det_mod):
        """.. method:: error(psi, det_mod)

            Calculates the MSE at a given position given the modes at that position.

            :param CXModal psi: A list of CXData instances containing all modes at a given position.
            :param np.ndarray det_mod: Modulus of measured diffraction pattern.

        """
        mode_sum = CXModal.modal_sum(abs(fft2(psi)))
        return (sp.sum((abs(mode_sum - det_mod) ** 2.).data[0]) / sp.sum(det_mod.data[0] ** 2.))**0.5


    def select_algorithm(self):
        try:
            self.algorithm_count
            self.ppc_count
        except AttributeError:
            self.algorithm_count = 0
            self.ppc_count = 0

        if self.algorithm == 'er':
            if self.algorithm_count>=CXP.reconstruction.algorithm['er']:
                self.algorithm = 'dm'
                self.algorithm_name = 'Difference Map'
                self.algorithm_count = 0
            else:
                self.algorithm_name = 'Error Reduction'
        elif self.algorithm == 'dm':
            if self.algorithm_count>=CXP.reconstruction.algorithm['dm']:
                self.algorithm = 'er'
                self.algorithm_name = 'Error Reduction'
                self.algorithm_count = 0
            else:
                self.algorithm_name = 'Difference Map'

        if self.total_its>CXP.reconstruction.ptycho_its-100:
            self.algorithm = 'er'
            self.algorithm_name = 'Error Reduction'

        if self.total_its>CXP.reconstruction.begin_updating_probe:# and self.algorithm=='er':
            self.do_update_probe = True
        else:
            self.do_update_probe=False

        if (CXP.reconstruction.probe_position_correction and 
            self.total_its>CXP.reconstruction.begin_probe_position_correction and
            self.ppc_count==0):
            self.ppc = True
            self.do_update_probe = False

        if self.ppc_count==1 and self.total_its>CXP.reconstruction.begin_updating_probe:
            self.do_update_probe = True

        self.ppc_count += 1
        if self.ppc_count>=5:
            self.ppc_count = 0


        if self.total_its==CXP.reconstruction.begin_modal_reconstruction:
            for m in range(CXP.reconstruction.probe_modes-1):
                self.probe.modes.append(CXData(data=0.05*self.probe.modes[0].data[0]))
                self.probe.orthogonalise()

        self.algorithm_count += 1
        self.db_queue['algorithm'] = (self.itnum, self.algorithm)

    def init_figure(self):
        pylab.ion()

        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)

        fig_width_pt  = 750 # Figure width in points

        inches_per_pt = 1.0 / 72.27
        golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

        fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
        fig_height_in = fig_width_in * golden_ratio   # figure height in inches
        fig_dims      = [fig_width_in, fig_height_in] # fig dims as a list

        self.fig = plt.figure(0, figsize = fig_dims)
        self.fig.canvas.set_window_title('CXPhasing: Ptychographic Phase Retrieval')

        try:
            thismanager = pylab.get_current_fig_manager()
            thismanager.window.wm_geometry("+600+0")
        except:
            pass

        try:
            itnum = self.itnum
        except AttributeError:
            itnum = 0
        try:
            mse = self.av_mse
        except AttributeError:
            mse = -1.0
        try:
            self.algorithm
        except AttributeError:
            self.algorithm = 'er'

        plt.suptitle('Scan: {:s}, Sequence: {:d}, Iteration: {:d}, MSE: {:3.2f}%, Algorithm: {:s}'.format(
            CXP.io.scan_id, CXP.reconstruction.sequence, itnum, 100*mse, self.algorithm))

        self.gs0 = gridspec.GridSpec(2,1)
        self.gs0.update(left=0.02, right=0.98, top=0.9, bottom=0.05, hspace=0.25)
        self.gs00 = gridspec.GridSpecFromSubplotSpec(60, 3, subplot_spec=self.gs0[0], hspace=0.0, wspace=0.1)
        self.gs01 = gridspec.GridSpecFromSubplotSpec(60, 12, subplot_spec=self.gs0[1], hspace=0.1, wspace=0.0)

        self.axes_group1 = []
        self.axes_group1.append(plt.Subplot(self.fig, self.gs00[0:58, 0])) #Image 1
        self.axes_group1.append(plt.Subplot(self.fig, self.gs00[58:59, 0])) #Colorbar 1
        self.axes_group1.append(plt.Subplot(self.fig, self.gs00[0:58, 1])) #Image 2
        self.axes_group1.append(plt.Subplot(self.fig, self.gs00[58:59, 1])) #Colorbar2
        self.axes_group1.append(plt.Subplot(self.fig, self.gs00[0:58, 2])) #Image 3
        self.axes_group1.append(plt.Subplot(self.fig, self.gs00[58:59, 2])) #Colorbar3

        [self.fig.add_subplot(ax) for ax in self.axes_group1]

        self.axes_group2 = []
        self.axes_group2.append(plt.Subplot(self.fig, self.gs01[58:59, 4:8])) # Colorbar
        if CXP.reconstruction.probe_modes==1:
            groups = [(0, 58, 0, 12)]
        elif CXP.reconstruction.probe_modes==2:
            groups = [(0, 58, 0, 6), (0, 58, 6, 12)]
        elif CXP.reconstruction.probe_modes==3:
            groups = [(0, 58, 0, 4), (0, 58, 4, 8), (0, 58, 8, 12)]
        elif CXP.reconstruction.probe_modes==4:
            groups = [(0, 58, 0, 3), (0, 58, 3, 6), (0, 58, 6, 9), (0, 58, 9, 12)]
        else:
            groups = [(0, 30, 0, 4),   (0, 30, 4, 8),  (0, 30, 8, 12),
                     (30, 58, 0, 4),  (30, 58, 4, 8), (30, 58, 8, 12)]

        for gr in groups:
            self.axes_group2.append(plt.Subplot(self.fig, self.gs01[gr[0]:gr[1], gr[2]:gr[3]])) #Image 1

        [self.fig.add_subplot(ax) for ax in self.axes_group2]


    def update_figure(self, i=0):

        [ax.cla() for ax in self.axes_group1]
        [ax.cla() for ax in self.axes_group2]

        wh = sp.where(abs(self.object.data[0]) > 0.1 * (abs(self.object.data[0]).max()))
        try:
            x1, x2 = min(wh[0]), max(wh[0])
            y1, y2 = min(wh[1]), max(wh[1])
        except (ValueError, IndexError):
            x1, x2 = 0, self.ob_p
            y1, y2 = 0, self.ob_p

        # Plot magnitude of object
        g1im1 = self.axes_group1[0].matshow(abs(self.object).data[0][x1:x2, y1:y2], cmap=cm.Greys_r)
        self.axes_group1[0].yaxis.set_major_locator(MaxNLocator(4))
        self.axes_group1[0].xaxis.set_major_locator(MaxNLocator(4))
        self.axes_group1[0].set_title(r'$|O|$')
        g1cb1 = Colorbar(ax = self.axes_group1[1], mappable = g1im1, orientation='horizontal')
        self.axes_group1[1].xaxis.set_major_locator(MaxNLocator(4))

        # Plot phase of object
        g1im2 = self.axes_group1[2].matshow(sp.angle(self.object.data[0][x1:x2, y1:y2]), cmap=cm.hsv)
        self.axes_group1[2].yaxis.set_major_locator(MaxNLocator(4))
        self.axes_group1[2].xaxis.set_major_locator(MaxNLocator(4))
        self.axes_group1[2].set_title(r'$-i\ln O$')
        g1cb2 = Colorbar(ax = self.axes_group1[3], mappable = g1im2, orientation='horizontal')

        # Complex HSV plot of object
        h = ((angle(self.object).data[0][x1:x2, y1:y2] + np.pi) / (2*np.pi)) % 1.0
        s = np.ones_like(h)
        l = abs(self.object).data[0][x1:x2, y1:y2]
        l-=l.min()
        l/=l.max()
        g1im3 = self.axes_group1[4].imshow(np.dstack(v_hls_to_rgb(h,l,s)))
        self.axes_group1[4].yaxis.set_major_locator(MaxNLocator(4))
        self.axes_group1[4].xaxis.set_major_locator(MaxNLocator(4))
        self.axes_group1[4].set_title(r'O')
        g1cb3 = Colorbar(ax = self.axes_group1[5], mappable = g1im3, orientation='horizontal')
        
        # Plot probe mode 0
        h = ((angle(self.probe[0]).data[0] + np.pi) / (2*np.pi)) % 1.0
        s = np.ones_like(h)
        l = abs(self.probe[0]).data[0]
        l-=l.min()
        l/=l.max()
        g2im1 = self.axes_group2[1].imshow(np.dstack(v_hls_to_rgb(h,l,s)))
        self.axes_group2[1].yaxis.set_major_locator(MaxNLocator(4))
        self.axes_group2[1].xaxis.set_major_locator(MaxNLocator(4))
        self.axes_group2[1].xaxis.set_ticks_position('top')
        self.axes_group2[1].set_title(r'Probe$_0$')
        g2cb = Colorbar(ax = self.axes_group2[0], mappable = g2im1, orientation='horizontal')

        if self.itnum>CXP.reconstruction.begin_modal_reconstruction:
            for mode in range(1, CXP.reconstruction.probe_modes):
                if mode<6:
                    h = ((angle(self.probe[mode]).data[0] + np.pi) / (2*np.pi)) % 1.0
                    s = np.ones_like(h)
                    l = abs(self.probe[mode]).data[0]
                    l-=l.min()
                    l/=l.max()
                    g2imN = self.axes_group2[1+mode].imshow(np.dstack(v_hls_to_rgb(h,l,s)))
                    self.axes_group2[1+mode].yaxis.set_major_locator(MaxNLocator(4))
                    self.axes_group2[1+mode].xaxis.set_major_locator(MaxNLocator(4))
                    self.axes_group2[1+mode].xaxis.set_ticks_position('top')
                    self.axes_group2[1+mode].set_title(r'Probe$_{:d}$'.format(mode))

        if self.ppc:
            try:
                self.fig2.clf()
            except AttributeError:
                self.fig2 = plt.figure(1)

            s6 = self.fig2.add_subplot(111)
            s6_im = s6.scatter(self.positions.data[0], self.positions.data[1], s=10,
                c='b', marker='o', alpha=0.5, edgecolors='none', label='current')
            patches = []
            for m in range(self.positions.total):
                patches.append(Circle((self.positions.initial[0][m], self.positions.initial[1][m]),
                               radius=CXP.reconstruction.ppc_search_radius))
            collection = PatchCollection(patches, color='tomato', alpha=0.2, edgecolors=None)
            s6.add_collection(collection)
            if CXP.measurement.simulate_data:
                s6_im = s6.scatter(self.positions.correct[0], self.positions.correct[1], s=10,
                    c='g', marker='o', alpha=0.5, edgecolors='none', label='correct')
                CXP.log.info('RMS position deviation from correct: [x:{:3.2f},y:{:3.2f}] pixels'.format(
                            sp.sqrt(sp.mean((self.positions.data[0] - self.positions.correct[0])**2.)),
                            sp.sqrt(sp.mean((self.positions.data[1] - self.positions.correct[1])**2.))))
                lines=[]
                for m in range(self.positions.total):
                    lines.append(((self.positions.correct[0][m], self.positions.correct[1][m]),
                                  (self.positions.data[0][m], self.positions.data[1][m])))
                for element in lines:
                    x, y = zip(*element)
                    s6.plot(x, y, 'g-')
            else:
                lines = []
                for m in range(self.positions.total):
                    lines.append(((self.positions.initial[0][m], self.positions.initial[1][m]),
                                  (self.positions.data[0][m], self.positions.data[1][m])))
                for element in lines:
                    x, y = zip(*element)
                    s6.plot(x, y, 'g-')
                CXP.log.info('RMS position deviation from initial: [x:{:3.2f},y:{:3.2f}] pixels'.format(
                            sp.sqrt(sp.mean((self.positions.data[0] - self.positions.initial[0])**2.)),
                            sp.sqrt(sp.mean((self.positions.data[1] - self.positions.initial[1])**2.))))
            s6.legend(prop={'size': 6})
            s6.set_title('Position Correction')
            s6.set_aspect('equal')
            extent = s6.get_window_extent().transformed(self.fig2.dpi_scale_trans.inverted())
            pylab.savefig(self._cur_sequence_dir + '/ppc_{:d}.png'.format(self.total_its), bbox_inches=extent.expanded(1.2, 1.2), dpi=100)
            s6.set_aspect('auto')

        pylab.draw()
        pylab.savefig(self._cur_sequence_dir + '/recon_{:d}.png'.format(self.total_its), dpi=60)

    def init_db_conn(self):

        # Make db connection
        self.db = SimpleDB()
        self.dbconn = self.db.conn

        # Select the CXParams db
        self.db.use(CXP.db.dbname)
        self.db.get_cursor()

        # Create table interface
        self.t_recon_id = self.db.tables['recon_id']
        self.t_recon_params = self.db.tables['recon_params']

        self.recon_id = self.t_recon_id.get_next_id(CXP.machine.name)
        CXP.log.info('MySQL Reconstruction ID: {}'.format(self.recon_id))

        # Add all of the static CXParams values
        for element in CXP.param_store.instances:
            for key, value in getattr(CXP, element).__dict__.iteritems():
                self.db_queue[key] = (-1, value)

    def update_recon_param_table(self):       

        then = time.time()
        cnt = 0
        for k, (itnum, v) in self.db_queue.iteritems():
            if isinstance(v, (list, tuple)):
                v='"{:s}"'.format(str(v))
            elif isinstance(v, (bool)):
                v=str(int(v))
            elif isinstance(v, str):
                v='"{:s}"'.format(v)
            self.t_recon_params.insert(recon_id=self.recon_id, iter=itnum,
                                        name='"{:s}"'.format(k), value=v)
            cnt+=1
        self.db_queue = {}
        now = time.time()
        self.db_queue['time_per_fast_db_entry'] = (self.itnum, (now - then) / cnt)
        CXP.log.info('{:3.2f} seconds elapsed entering {:d} values into fast db [{:3.2f} msec/entry]'.format(now-then,
                        cnt, 1e3 * (now - then) / cnt))

    def log_reconstruction_parameters(self):
        """
        h - object size\nz - sam-det dist\npix - # of pix\ndel_x_d - pixel size
        """
        dx_d = CXP.experiment.dx_d
        x = (CXP.p/2.)*dx_d
        l = energy_to_wavelength(CXP.experiment.energy)
        h = min(CXP.experiment.beam_size)
        pix = CXP.p
        z=CXP.experiment.z
        NF = lambda nh, nl, nz: nh**2./(nl*nz)
        del_x_s = lambda l, z, x: (l*z)/(2.*x)
        nNF = NF(h, l, z)
        OS = lambda l, z, x, h, pix: ((pix*del_x_s(l, z, x))**2.)/(h**2.)
        nOS = OS(l, z, x, h, pix)
        NA = sp.sin(sp.arctan(x/z))
        axial_res = 2*l/NA**2.
        lateral_res = l/(2.*NA)
        CXP.log.info('Fresnel number: {:2.2e}'.format(nNF))
        CXP.log.info('Oversampling: {:3.2f}'.format(nOS))
        CXP.log.info('Detector pixel size: {:3.2f} [micron]'.format(1e6*dx_d))
        CXP.log.info('Detector width: {:3.2f} [mm]'.format(1e3*pix*dx_d))
        CXP.log.info('Sample pixel size: {:3.2f} [nm]'.format(1e9*del_x_s(l, z, x)))
        CXP.log.info('Sample FOV: {:3.2f} [micron]'.format(1e6*del_x_s(l, z, x)*pix))
        CXP.log.info('Numerical aperture: {:3.2f}'.format(NA))
        CXP.log.info('Axial resolution: {:3.2f} [micron]'.format(1e6*axial_res))
        CXP.log.info('Lateral resolution: {:3.2f} [nm]'.format(1e9*lateral_res))

        self.db_queue['fresnel_number'] = (-1, nNF)
        self.db_queue['oversampling'] = (-1, nOS)
        self.db_queue['dx_s'] = (-1, del_x_s(l, z, x))
        self.db_queue['sample_fov'] = (-1, del_x_s(l, z, x)*pix)
        self.db_queue['numerical_aperture'] = (-1, NA)
        self.db_queue['axial_resolution'] = (-1, axial_res)

    def setup_dir_tree(self):
        """Setup the directory structure for a new scan id"""
        _top_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id])
        _sequence_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'sequences'])
        _cur_sequence_dir = _sequence_dir+'/sequence_{:d}'.format(CXP.reconstruction.sequence)
        _raw_data_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'raw_data'])
        _dpc_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'dpc'])
        _CXP_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, '.CXPhasing'])
        _py_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'python'])

        if not os.path.exists(_top_dir):
            CXP.log.info('Setting up new scan directory...')
            os.mkdir(_top_dir)
            os.mkdir(_sequence_dir)
            os.mkdir(_cur_sequence_dir)
            os.mkdir(_raw_data_dir)
            os.mkdir(_dpc_dir)
            os.mkdir(_CXP_dir)
            os.mkdir(_py_dir)
            try:
                shutil.copy(CXP.io.code_dir+'cxparams/CXParams.py', _py_dir)
            except IOError:
                CXP.log.error('Was unable to save a copy of CXParams.py to {}'.format(_py_dir))
        else:
            CXP.log.info('Dir tree already exists.')
            if not os.path.exists(_sequence_dir):
                os.mkdir(_sequence_dir)
            if not os.path.exists(_cur_sequence_dir):
                CXP.log.info('Making new sequence directory')
                os.mkdir(_cur_sequence_dir)
            try:
                shutil.copy(CXP.io.code_dir+'cxparams/CXParams.py', _py_dir)
                shutil.copy(CXP.io.code_dir+'cxparams/CXParams.py',
                            _cur_sequence_dir+'/CXParams_sequence{}.py'.format(CXP.reconstruction.sequence))
            except IOError:
                CXP.log.error('Was unable to save a copy of CXParams.py to {}'.format(_py_dir))

    def ptycho_mesh(self):
        """
        Generate a list of ptycho scan positions.

        Outputs
        -------
        self.positions.data : list of 2xN arrays containing horizontal and vertical scan positions in pixels
        self.positons.initial : initial guess at ptycho scan positions (before position correction)
        self.positions.total : total number of ptycho positions

        [optional]
        self.positions.correct : for simulated data this contains the correct position

        """
        CXP.log.info('Getting ptycho position mesh.')
        
        if CXP.measurement.ptycho_scan_mesh == 'generate':
            if CXP.measurement.ptycho_scan_type == 'cartesian':
                x2 = 0.5*(CXP.measurement.cartesian_scan_dims[0]-1)
                y2 = 0.5*(CXP.measurement.cartesian_scan_dims[1]-1)
                tmp = map(lambda a: CXP.measurement.cartesian_step_size*a, np.mgrid[-x2:x2+1, -y2:y2+1])
                self.positions.data = [tmp[0].flatten(), tmp[1].flatten()]
                if CXP.reconstruction.flip_mesh_lr:
                    CXP.log.info('Flip ptycho mesh left-right')
                    self.positions.data[0] = self.data[0][::-1]
                if CXP.reconstruction.flip_mesh_ud:
                    CXP.log.info('Flip ptycho mesh up-down')
                    self.positions.data[1] = self.data[1][::-1]
                if CXP.reconstruction.flip_fast_axis:
                    CXP.log.info('Flip ptycho mesh fast axis')
                    tmp0, tmp1 = self.positions.data[0], self.positions.data[1]
                    self.positions.data[0], self.positions.data[1] = tmp1, tmp0
            if CXP.measurement.ptycho_scan_type == 'round_roi':
                self.positions.data = list(round_roi(CXP.measurement.round_roi_diameter, CXP.measurement.round_roi_step_size))
            if CXP.measurement.ptycho_scan_type == 'list':
                l = np.genfromtxt(CXP.measurement.list_scan_filename)
                x_pos, y_pos = [], []
                for element in l:
                    x_pos.append(element[0])
                    y_pos.append(element[1])
                self.positions.data = [sp.array(x_pos), sp.array(y_pos)]


        elif CXP.measurement.ptycho_scan_mesh == 'supplied':
            l = np.genfromtxt(CXP.measurement.list_scan_filename)
            x_pos, y_pos = [], []
            for element in l:
                x_pos.append(element[0])
                y_pos.append(element[1])
            self.positions.data = [sp.array(x_pos), sp.array(y_pos)]

        for element in self.positions.data:
            element /= CXP.dx_s
            element += CXP.ob_p/2
        self.positions.total = len(self.positions.data[0])

        self.positions.correct = [sp.zeros((self.positions.total))]*2
        jit_pix = CXP.reconstruction.initial_position_jitter_radius
        search_pix = CXP.reconstruction.ppc_search_radius

        self.positions.data[0] += jit_pix * uniform(-1, 1, self.positions.total)
        self.positions.data[1] += jit_pix * uniform(-1, 1, self.positions.total)
        self.positions.initial = [self.positions.data[0].copy(), self.positions.data[1].copy()]

        if CXP.reconstruction.probe_position_correction:
            self.positions.correct[0] = self.positions.data[0]+0.25*search_pix * uniform(-1, 1, self.positions.total)
            self.positions.correct[1] = self.positions.data[1]+0.25*search_pix * uniform(-1, 1, self.positions.total)
        else:
            self.positions.correct = [self.positions.data[0].copy(), self.positions.data[1].copy()]

        data_copy = CXData(data=list(self.positions.data))
        if not CXP.reconstruction.ptycho_subpixel_shift:
            self.positions.data = [np.round(self.positions.data[0]), np.round(self.positions.data[1])]
            self.positions.correct = [np.round(self.positions.correct[0]), np.round(self.positions.correct[1])]
        CXP.rms_rounding_error = [None]*2

        for i in range(2):
            CXP.rms_rounding_error[i] = sp.sqrt(sp.sum(abs(abs(data_copy.data[i])**2.-abs(self.positions.data[i])**2.)))

        CXP.log.info('RMS Rounding Error (Per Position, X, Y):\t {:2.2f}, {:2.2f}'.format(CXP.rms_rounding_error[0]/len(self.positions.data[0]),
                                                                                           CXP.rms_rounding_error[1]/len(self.positions.data[1])))

    def init_probe(self, *args, **kwargs):

        if CXP.io.initial_probe_guess is not '':
            probe = CXData()
            probe.load(CXP.io.initial_probe_guess)
            self.probe.modes = [CXData(data=[probe.data[0]/(i+1)]) for i in range(CXP.reconstruction.probe_modes)]
            self.probe.normalise()
        else:

            dx_s = CXP.dx_s

        p, p2 = CXP.preprocessing.desired_array_shape, CXP.preprocessing.desired_array_shape/2

        probe = sp.zeros((p, p), complex)

        if CXP.experiment.optic.lower() == 'kb':
            if len(CXP.experiment.beam_size)==1:
                bsx=bsy=np.round(CXP.experiment.beam_size[0]/dx_s)
            elif len(CXP.experiment.beam_size)==2:
                bsx, bsy = np.round(CXP.experiment.beam_size[0]/dx_s), np.round(CXP.experiment.beam_size[1]/dx_s)

            probe = np.sinc((np.arange(p)-p2)/bsx)[:,np.newaxis]*np.sinc((np.arange(p)-p2)/bsy)[np.newaxis,:]
            

        elif CXP.experiment.optic.lower() == 'zp':
            probe = np.sinc(sp.hypot(*sp.ogrid[-p2:p2, -p2:p2])/np.round(3.*CXP.experiment.beam_size[0]/(2*CXP.dx_s)))

        ph_func = gauss_smooth(np.random.random(probe.shape), 10)
        fwhm = p/2.0
        radker = sp.hypot(*sp.ogrid[-p/2:p/2,-p/2:p/2])
        gaussian = exp(-1.0*(fwhm/2.35)**-2. * radker**2.0 )
        gaussian /= gaussian.max()
        if CXP.measurement.simulate_data:
            probe = abs(gaussian*probe)* exp(complex(0.,np.pi)*ph_func/ph_func.max())
        else:
            probe = abs(gaussian*probe).astype(complex)

        avdata = reduce(np.add, self.det_mod.data)/len(self.det_mod)

        probe  = ifft2(avdata * exp(complex(0.,1.)*sp.angle(fft2(probe))))

        self.probe.modes = [CXData(data=[probe])]
        
        self.probe.normalise()

    def calc_stxm_image(self):
        path = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'sequences/sequence_{:d}/stxm_regular_grid.png'.format(CXP.reconstruction.sequence)])
        CXP.log.info('Calculating STXM image.\nSTXM saved to:\n\t{}'.format(path))
        image_sum = sp.array([sp.sum(data) for data in self.det_mod.data])

        x, y = self.positions.data

        fig = Figure(figsize=(6, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title('STXM Image', fontsize=14)
        ax.set_xlabel('Position [micron]', fontsize=12)
        ax.set_ylabel('Position [micron]', fontsize=12)

        if CXP.measurement.ptycho_scan_type == 'cartesian':
            ax.hexbin(x, y, C=image_sum, gridsize=CXP.measurement.cartesian_scan_dims, cmap=cm.RdGy)
            canvas.print_figure('/'.join([CXP.io.base_dir, CXP.io.scan_id, 'sequences/sequence_{:d}/stxm_scatter.png'.format(CXP.reconstruction.sequence)]), dpi=500)
            ax.imshow(image_sum.reshape(CXP.measurement.cartesian_scan_dims), cmap=cm.RdGy)
        else:
            ax.hexbin(x, y, C=image_sum, cmap=cm.RdGy)

        canvas.print_figure(path, dpi=500)

    def maximum_likelihood_refinement(self):

        """
        ..method:: maxmimum_likelihood_refinement(self)
            Implements the MLE method of Thibault & Guizar-Sicairos, New. J. Phys. 14 063004 (2012).

        """
        pdb.set_trace()
        probe = CXData()
        ob = CXData()
        det_mod = CXData()
        probe.load('/home/david/data/sec34Robinson/S021/mle/probe_mode0.npz')
        ob.load('/home/david/data/sec34Robinson/S021/mle/object.npz')
        det_mod.load('/home/david/data/sec34Robinson/S021/mle/det_mod.npz')

        scalar_product = lambda a, b: CXData.inner_product(a[0],b[0])+CXData.inner_product(a[1],b[1])

        self.positions = CXData(name='positions', data=[])
        self.ptycho_mesh()
        self.p2 = CXP.preprocessing.desired_array_shape

        pdb.set_trace()

        for r in range(CXP.mle.its):
            gn = self.calc_derivatives(probe, ob, det_mod)

            if r==0:
                beta = 0
                delta_old = 0
            else:
                beta = (scalar_product(gn, gn) - scalar_product(gn, gn_old))/scalar_product(gn_old, gn_old)

            delta = -gn + beta*delta_old

            coef = sp.zeros((9))

            for i in range(self.positions.total):

                a = fft2( self.probe*delta[0][d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2] +
                          self.object[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]*delta[1] )

                b = fft2(delta[0][d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]*delta[1])

                psi_bar =  fft2(self.probe * self.object[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2])

                A0 = abs(psi_bar)**2.0 - det_mod

                A1 = 2.0*np.real(psi_bar*conj(a))

                A2 = 2.0*np.real(psi_bar*conj(b)) + abs(a)**2.0

                A3 = 2.0*np.real(a*conj(b))

                A4 = abs(b)**2.0

                coef[8]+=sum((A0**2.0)/(det_mod+1))
                coef[7]+=sum((2*A0*A1)/(det_mod+1))
                coef[6]+=sum((2*A0*A2+A1**2.0)/(det_mod+1))
                coef[5]+=sum((2*A0*A3+2*A1*A2)/(det_mod+1))
                coef[4]+=sum((2*A0*A4+2*A1*A3+A2**2.0)/(det_mod+1))
                coef[3]+=sum((2*A1*A4+2*A1*A3)/(det_mod+1))
                coef[2]+=sum((2*A2*A4+A3**2)/(det_mod+1))
                coef[1]+=sum((2*A3*A4)/(det_mod+1))
                coef[0]+=sum((A4**2)/(det_mod+1))

            dcoefs = [(8-i)*coef[i] for i in range(0,8)]

            rts = np.roots(dcoef)

            lhoods = [self.gaussian_likelihood(element) for element in rts]
            gamma = rts[lhoods.index(lhoods.min())]

            ob += gamma*delta[0]
            probe += gamma*delta[1]

            gn_old = [gn[0].copy(), gn[1].copy()]
            delta_old = [delta[0].copy(), delta[1].copy()]

    def gaussian_likelihood(self, probe, ob):

        d1, d2 = self.positions.data
        likelihood = 0
        for i in range(self.position.total):
            view = probe * ob[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]
            likelihood += sum((abs(fft2(view))**2.0 - det_mod[i]**2.0)**2.0/(2*(det_mod[i]+1.0)))

    def calc_derivatives(self, probe, ob, det_mod):
        chi = lambda view, dm: ifft2(((abs(fft2(view))-dm)/(dm+1.0))*fft2(view))
        gor = sp.zeros((self.ob_p, self.ob_p),complex)
        gpr = sp.zeros((self.ob_p, self.ob_p),complex)
        d1, d2 = self.positions.data
        id1, id2 = d1//1, d2//1
        for i in range(self.positions.total):
            view = probe * ob[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]
            for mode in range(CXP.reconstruction.probe_modes):
                gor[id1[i] - self.p2:id1[i] + self.p2, id2[i] - self.p2:id2[i] + self.p2]+= probe[mode][0]*conj(chi(view[mode][0], det_mod[i]))
                gpr += ob[id1[i] - self.p2:id1[i] + self.p2, id2[i] - self.p2:id2[i] + self.p2] *conj(chi(view[mode][0], det_mod[i]))
        
        return (gor, gpr)

    def db_plot(self):
        pdb.set_trace()

        n_names = len(self.t_recon_params.select(vals='distinct name', 
                        where='recon_id={:d} and iter>-1'.format(self.recon_id)))
        res = self.t_recon_params.select(vals='name, value, iter', 
                       where='recon_id={:d} and iter>-1'.format(self.recon_id))

        names = []
        enum = {}
        for entry in res:
            if isinstance(entry['value'], str):
                if entry['name'] in enum.keys():
                    if entry['value'] in enum[entry['name']]:
                        pass
                    else:
                        enum[entry['name']].append(entry['value'])
                else:
                    enum[entry['name']] = [entry['value']]

        data = sp.zeros((n_names, self.itnum))
        for entry in res:
            if entry['name'] not in names:
                names.append(entry['name'])
            idx = names.index(entry['name'])
            if isinstance(entry['value'], str):
                entry['value'] = enum[entry['name']].index(entry['value'])
            data[idx, entry['iter']] = entry['value']

        db_fig = plt.figure()

        for i in range(n_names):
            s=plt.subplot('{0:d}1{0:d}'.format(i))
            s.plot(data[i,:], label=names[i])
            s.set_title(names[i])
        pylab.savefig(self._cur_sequence_dir + '/db_plot.png', dpi=100)
