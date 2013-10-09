"""
.. module:: CXPhasing2.py
   :platform: Unix
   :synopsis: Implements phase retrieval algorithms.

.. moduleauthor:: David Vine <djvine@gmail.com>


""" 

import os
import numpy as np
import scipy as sp
import pylab
import time
import math
import pdb
from numpy.random import uniform
import multiprocessing as mp
import itertools
import shutil

import cxphasing.cxparams.CXParams as CXP
from CXData2 import fft2, ifft2, angle, exp, fftshift, conj, abs, sqrt
from CXData2 import log as nlog
from CXData2 import CXData, CXModal
from CXDb import SimpleDB
from CXUtils import worker, object_worker, split_seq, v_hls_to_rgb, energy_to_wavelength, gauss_smooth, tukeywin
import multiprocess

from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

try:
    import MySQLdb
    hasmysql = True
except:
    hasmysql = False

class CXPhasing(object):
    """
    .. class:: CXPhasing(object)
        Implements phase retrieval process.


        :attr annealing_schedule: Annealing schedule for probe position correction
        :type annealing_schedule: lambda function
        :attr dict slow_db_queue: 
            Values to be entered into the slow (once per reconstruction attempt) database.
               Entry syntax:
               slow_db_queue[db_field] = (value, )
        :attr dict fast_db_queue: 
            Values to be entered into the fast (once per iteration per reconstruction attempt) database.
            Entry syntax:
                fast_db_queue[db_field] = (iter, value)
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

        self.ppc = CXP.reconstruction.probe_position_correction

        # MySQL DB Integration
        if hasmysql:
            self.init_db_conn()
        # Values are inserted into the db by adding them to the queue
        # The queues are emptied once per iteration
        # The slow database has one entry per reconstruction attempt
        # The fast database has one entry per iteration per reconstruction attempt
        # Entry syntax:
        #   slow_db_queue[db_field] = (value, )
        #   fast_db_queue[db_field] = (iter, value)
        self.slow_db_queue = {}
        self.fast_db_queue = {}

        self.p = CXP.p
        self.p2 = self.p / 2
        self.ob_p = CXP.preprocessing.object_array_shape
        self.total_its = 0
        self.probe_modes = CXP.reconstruction.probe_modes

        self.algorithm = 'er' # Start with error reduction

        if CXP.machine.n_processes < 0:
            CXP.machine.n_processes = mp.cpu_count()

        self.epie_repr = '{:s}\n\tPtychography iteration:{:10d}\n\tPtychography position:{:10d} [{:3.0f}%]'
        self.progress_repr = 'Current iteration: {:d}\tPosition: {:d}'
        
        self._sequence_dir = '/'.join([CXP.io.base_dir, CXP.io.scan_id, 'sequences'])
        self._cur_sequence_dir = self._sequence_dir+'/sequence_{:d}'.format(CXP.reconstruction.sequence)

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
            else:
                self.det_mod.load()
            if CXP.io.whitefield_filename:
                self.probe_det_mod = CXData(name='probe_det_mod')
                self.probe_det_mod.preprocess_data()

        self.object = CXData(name='object', data=[sp.zeros((self.ob_p, self.ob_p), complex)])
        
        self.probe_intensity = CXData(name='probe_intensity', data=[sp.zeros((self.p, self.p))])

        self.probe = CXModal(modes=[])
        self.psi = CXModal(modes=[])

        for i in range(CXP.reconstruction.probe_modes):
            self.probe.modes.append(CXData(name='probe{:d}'.format(i), data=[sp.zeros((self.p, self.p), complex)]))
            self.psi.modes.append(CXData(name='psi{:d}'.format(i), data=[sp.zeros((self.p, self.p), complex) for i in xrange(self.det_mod.len())]))
    
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
        
        if hasmysql:
            self.update_slow_table()
        beginning = time.time()
        
        for self.itnum in xrange(its):
            then = time.time()

            self.select_algorithm()

            self.ePIE()

            now = time.time()
            if hasmysql:
                self.fast_db_queue['iter_time'] = (self.itnum, now - then)
                self.fast_db_queue['iter_time_pptpxit'] = (self.itnum, 1e6*(now - then) / (self.positions.total * self.p**2 * (self.itnum + 1)))
            CXP.log.info('{:2.2f} seconds elapsed during iteration {:d} [{:1.2e} sec/pt/pix/it]'.format(now - then, self.itnum + 1,
                            (now-then)/(self.positions.total * self.p**2 * (self.itnum + 1))))
            CXP.log.info('{:5.2f} seconds have elapsed in {:d} iterations [{:2.2f} sec/it]'.format(now-beginning, self.itnum + 1, (now-beginning)/(self.total_its + 1)))
            self.calc_mse()
            self.total_its += 1
            if hasmysql:
                self.update_fast_table()
            if self.itnum > 0:
                self.update_figure(self.itnum)

    def postprocessing(self):
        """.. method::postprocessing()
            Collectes together all the orutines that should be completed after the iterative phase retrieval has successfully completed.

        """
        pass

    def simulate_data(self):
        CXP.log.info('Simulating diffraction patterns.')
        self.sample = CXData()
        self.sample.load(CXP.io.simulation_sample_filename[0])
        self.sample.data[0] = self.sample.data[0].astype(float)
        self.sample.normalise(val=0.8)
        self.sample.data[0]+=0.2
        self.input_probe = CXModal()
        if len(CXP.io.simulation_sample_filename)>1:
            ph = CXData()
            ph.load(CXP.io.simulation_sample_filename[1])
            ph.data[0] = ph.data[0].astype(float)
            ph.normalise(val=np.pi/3)
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
            else:
                new_mode = list(mode_generator())
                while new_mode in used_modes:
                    new_mode = list(mode_generator())
            used_modes.append(new_mode)
            CXP.log.info('Simulating mode {:d}: [{:d}, {:d}]'.format(mode, int(new_mode[0]), int(new_mode[1])))
            ph_func = gauss_smooth(np.random.random((p,p)), 10)
            self.input_probe.modes.append(CXData(name='probe{:d}'.format(mode), 
                data=ortho_modes(new_mode[0], new_mode[1])*exp(complex(0.,np.pi)*ph_func/ph_func.max())))
        
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
        self.det_mod.save(path=CXP.io.base_dir+'/'+CXP.io.scan_id+'/raw_data/{:s}.npy'.format('det_mod'))

    def pos_correction_transform(self, i, itnum):
        # Generates trial position
        search_rad = CXP.reconstruction.ppc_search_radius

        r = self.annealing_schedule(itnum)

        cx = self.positions.data[0][i] + (search_rad * r * uniform(-1, 1))
        cy = self.positions.data[1][i] + (search_rad * r * uniform(-1, 1))

        # Limit max deviation
        if np.abs(cx - self.positions.initial[0][i]) > search_rad:
            cx = self.positions.initial[0][i] + search_rad * r * uniform(-1, 1)
        if np.abs(cy - self.positions.initial[1][i]) > search_rad:
            cy = self.positions.initial[1][i] + search_rad * r * uniform(-1, 1)

        if CXP.reconstruction.ptycho_subpixel_shift:
            return [cx, cy]
        else:
            return [np.round(cx), np.round(cy)]

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
            mode_sum = CXModal.modal_sum(abs(fft2(psi))**2.0)**0.5
            return ifft2((fft2(psi)/(mode_sum))*det_mod)

    def ePIE(self):
        """.. method:: initial_update_state_vector(self)

            This method uses ePie to generate the initial estimate for psi and object.

        """

        d1, d2 = self.positions.data
        for i in xrange(self.positions.total):

            if i % np.floor(self.positions.total / 10) == 0 and CXP.reconstruction.verbose:
                CXP.log.info(self.epie_repr.format(self.algorithm_name, self.itnum, i, 100. * float(i + 1) / self.positions.total))

            # Non-modal reconstruction
            if self.total_its<CXP.reconstruction.begin_modal_reconstruction: 

                if self.itnum+i==0:
                    view=self.probe[0][0].copy()
                else:
                    view = self.probe[0][0] * self.object[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]
                
                if self.algorithm == 'er':
                    self.psi[0][i] = self.M(view.copy(), self.det_mod[i])
                elif self.algorithm == 'dm':
                    self.psi[0][i] += self.M(2*view-self.psi[0][i], self.det_mod[i]) - view
                    
                self.update_object(i, view, self.psi[0][i])
                if self.do_update_probe:
                    self.update_probe_nonmodal(i, view, self.psi[0][i])
            
            else: # Do modal reconstruction 
                view = self.probe * self.object[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]

                if self.algorithm == 'er':
                    self.psi.setat(i, self.M(view, self.det_mod[i]))
                    
                elif self.algorithm == 'dm':
                    self.psi.setat(i, self.psi.getat(i)+self.M(2*view-self.psi, self.det_mod[i]) - view)

                self.update_object(i, view, self.psi.getat(i))
                if self.do_update_probe:
                    self.update_probe(i, view, self.psi.getat(i))

        for mode, probe in enumerate(self.probe.modes):
            probe.save(path=self._cur_sequence_dir+'/probe_mode{:d}'.format(mode))
        self.object.save(path=self._cur_sequence_dir+'/object')

    def update_object(self, i, psi_old, psi_new):
        """
        Update the object from a single ptycho position.

        """
        then=time.time()
        d1, d2 = self.positions.data
        id1, id2 = d1//1, d2//1
        probe_intensity_max = CXModal.modal_sum(abs(self.probe)**2.0).data[0].max()
        
        self.object[id1[i] - self.p2:id1[i] + self.p2, id2[i] - self.p2:id2[i] + self.p2] += \
            CXData.shift(CXModal.modal_sum(conj(self.probe) * (psi_new-psi_old)) / probe_intensity_max, 
                d1[i]%1, d2[i]%1)

        if self.total_its==0 and sp.mod(i, len(self.positions.data[0]) / 10) == 0:
            self.update_figure(i)

    def update_probe_nonmodal(self, i, psi_old, psi_new):
        
        d1, d2 = self.positions.data
        id1, id2 = d1//1, d2//1
        object_intensity_max = (abs(self.object)**2.0).data[0].max()

        self.probe.modes[0] += \
            CXData.shift(conj(self.object[id1[i] - self.p2:id1[i] + self.p2, id2[i] - self.p2:id2[i] + self.p2]) *
             (psi_new-psi_old)[0] / object_intensity_max, d1[i]%1, d2[i]%1)

        self.probe.normalise()

    def update_probe(self, i, psi_old, psi_new):
        
        d1, d2 = self.positions.data
        id1, id2 = d1//1, d2//1
        object_intensity_max = (abs(self.object)**2.0).data[0].max()

        for mode in range(len(self.probe)):
            self.probe.modes[mode] += \
                CXData.shift(conj(self.object[id1[i] - self.p2:id1[i] + self.p2, id2[i] - self.p2:id2[i] + self.p2]) *
                 (psi_new-psi_old)[mode] / object_intensity_max, d1[i]%1, d2[i]%1)

        self.probe.normalise()
        
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
        except AttributeError:
            self.algorithm_count = 0

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

        self.algorithm_count += 1
        self.fast_db_queue['algorithm'] = (self.itnum, self.algorithm)

    def init_figure(self):
        pylab.ion()
        self.f1=pylab.figure(1, figsize=(12, 10))
        thismanager = pylab.get_current_fig_manager()
        thismanager.window.wm_geometry("+600+0")
        try:
            itnum = self.itnum
        except AttributeError:
            itnum = 0
        try:
            mse = self.av_mse
        except AttributeError:
            mse = -1.0
        pylab.suptitle('Sequence: {:d}, Iteration: {:d}, MSE: {:3.2f}%'.format(CXP.reconstruction.sequence, itnum, 100*mse))


    def update_figure(self, i=0):
        cur_cmap = cm.RdGy_r
        self.f1.clf()
        self.init_figure()

        wh = sp.where(abs(self.object.data[0]) > 0.1 * (abs(self.object.data[0]).max()))
        try:
            x1, x2 = min(wh[0]), max(wh[0])
            y1, y2 = min(wh[1]), max(wh[1])
        except (ValueError, IndexError):
            x1, x2 = 0, self.ob_p
            y1, y2 = 0, self.ob_p

        # Plot magnitude of object
        s1 = pylab.subplot(231)
        s1_im = s1.imshow(abs(self.object).data[0][x1:x2, y1:y2], cmap=cm.Greys_r)
        s1.set_title('|object|')
        plt.axis('off')
        pylab.colorbar(s1_im)

        # Plot phase of object
        s2 = pylab.subplot(232)
        s2_im = s2.imshow(sp.angle(self.object.data[0][x1:x2, y1:y2]), cmap=cm.hsv)
        s2.set_title('phase(object)')
        plt.axis('off')
        pylab.colorbar(s2_im)

        # Complex HSV plot of object
        s3 = pylab.subplot(233)
        h = ((angle(self.object).data[0][x1:x2, y1:y2] + np.pi) / (2*np.pi)) % 1.0
        s = np.ones_like(h)
        l = abs(self.object).data[0][x1:x2, y1:y2]
        l-=l.min()
        l/=l.max()
        s3_im = s3.imshow(np.dstack(v_hls_to_rgb(h,l,s)))
        s3.set_title('Complex plot of Object')
        plt.axis('off')

        # Plot probe mode 0
        s4 = pylab.subplot(234)
        s4_im = s4.imshow(abs(self.probe.modes[0].data[0]), cmap=cur_cmap)
        s4.set_title('|probe0|')
        plt.axis('off')
        pylab.colorbar(s4_im)

        if CXP.reconstruction.probe_modes>1:
            s5 = pylab.subplot(235)
            s5_im = s5.imshow(abs(self.probe.modes[1].data[0]), cmap=cur_cmap)
            s5.set_title('|probe1|')
            plt.axis('off')
            pylab.colorbar(s5_im)
        else:
            pass
        if self.ppc:
            s6 = self.f1.add_subplot(236)
            s6_im = s6.scatter(self.positions.data[0], self.positions.data[1], s=10,
                c='b', marker='o', alpha=0.5, edgecolors='none', label='current')
            patches = []
            for m in range(self.positions.total):
                patches.append(Circle((self.positions.initial[0][m], self.positions.initial[1][m]),
                               radius=CXP.reconstruction.ppc_search_radius))
            collection = PatchCollection(patches, color='tomato', alpha=0.2, edgecolors=None)
            s4.add_collection(collection)
            if CXP.measurement.simulate_data:
                s4_im = s4.scatter(self.positions.correct[0], self.positions.correct[1], s=10,
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
                    s4.plot(x, y, 'g-')
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
            extent = s6.get_window_extent().transformed(self.f1.dpi_scale_trans.inverted())
            pylab.savefig(self._cur_sequence_dir + '/ppc_{:d}.png'.format(self.total_its), bbox_inches=extent.expanded(1.2, 1.2), dpi=100)
            s6.set_aspect('auto')
        else:
            s6 = pylab.subplot(236)
            if CXP.measurement.simulate_data:
                s6_im = s6.imshow(abs(self.input_probe[1].data[0]), cmap = cur_cmap)
                s6.set_title('|input_probe1|')
            else:
                s6_im = s6.imshow(nlog(fftshift(self.det_mod[np.mod(i,self.positions.total)])).data[0], cmap=cur_cmap)
                s6.set_title('Diff Patt: {:d}'.format(i))
            plt.axis('off')
            pylab.colorbar(s6_im)
        pylab.draw()
        pylab.savefig(self._cur_sequence_dir + '/recon_{:d}.png'.format(self.total_its), dpi=60)

    def init_db_conn(self):

        # Make db connection
        self.db = SimpleDB()
        self.dbconn = self.db.conn

        # Select the CXParams db
        self.db.use(CXP.db.master_db)
        self.db.get_cursor()

        # Create table interface
        self.t_slow_params = self.db.tables['slow_params']
        self.t_fast_params = self.db.tables['fast_params']

        self.recon_id = self.t_slow_params.get_new_recon_id()
        CXP.log.info('MySQL Reconstruction ID: {}'.format(self.recon_id))

    def update_slow_table(self):

        for element in CXP.param_store.instances:
            for key, value in getattr(CXP, element).__dict__.iteritems():
                self.slow_db_queue[key] = (value,)
        
        then = time.time()
        cnt = 0
        for k, (v,) in self.slow_db_queue.iteritems():
            if isinstance(v, (list, tuple)):
                v=str(v)
            self.t_slow_params.insert_on_duplicate_key_update(primary={'id': self.recon_id}, update={k: v})
            cnt += 1
        now = time.time()
        self.slow_db_queue['time_per_slow_db_entry'] = (now - then)/cnt
        CXP.log.info('{:3.2f} seconds elapsed entering {:d} values into slow db [{:3.2f} msec/entry]'.format(now-then,
                        cnt, 1e3*(now - then) / cnt))

    def update_fast_table(self):

        if not self.t_fast_params.check_columns(self.fast_db_queue.keys()):
            for key, (itnum, value) in self.fast_db_queue.iteritems():
                if not self.t_fast_params.check_columns([key]):
                    CXP.log.warning('MYSQL: Adding column {} to fast_params.'.format(key))
                    ftype = 'double'
                    if isinstance(value, (list, tuple)):
                        value = str(value)
                    if isinstance(value, str):
                        ftype = 'text'
                        def_val = ''
                    elif isinstance(value, bool):
                        ftype = 'bool'
                        def_val = ''
                    elif isinstance(value, (int, float)):
                        ftype = 'double'
                        def_val = 0
                    else:
                        ftype = 'blob'
                        def_val = ''
                    self.t_fast_params.add_column(col_name=key, var_type=ftype, default_value=def_val)
            self.t_fast_params.update_fieldtypes()

        then = time.time()
        cnt = 0

        for k, (itnum, v) in self.fast_db_queue.iteritems():
            if isinstance(v, (list, tuple)):
                v=str(v)
            self.t_fast_params.insert_on_duplicate_key_update(
                primary={'slow_id': self.recon_id, 'iter': itnum}, update={k: v})
            cnt+=1
        now = time.time()
        self.fast_db_queue['time_per_fast_db_entry'] = (self.itnum, (now - then) / cnt)
        CXP.log.info('{:3.2f} seconds elapsed entering {:d} values into fast db [{:3.2f} msec/entry]'.format(now-then,
                        cnt, 1e3 * (now - then) / cnt))

    def calc_mse(self):
        then = time.time()

        multip = multiprocess.multiprocess(self.mse_worker)

        d1, d2 = self.positions.data

        for i_range in list(split_seq(range(self.positions.total),
                CXP.machine.n_processes)):
                multip.add_job((i_range, self.psi, self.det_mod))

        results = multip.close_out()

        self.av_mse = sp.mean(list(itertools.chain(*results)))

        CXP.log.info('Mean square error: {:3.2f}%'.format(100 * self.av_mse))
        self.fast_db_queue['error'] = (self.itnum, self.av_mse)
        now = time.time()
        CXP.log.info('Calculating MSE took {:3.2f}sec [{:3.2f}msec/position]'.format(now - then,
                       1e3*(now - then) / self.positions.total))

    @staticmethod
    @multiprocess.worker
    def mse_worker(args):
        i_range, psi, det_mod = args
        indvdl_mse = []
        p = det_mod[0].data[0].shape[0]
        for i in i_range:
            psi_sum = CXModal.modal_sum(abs(fft2(psi.getat(i))))
            indvdl_mse.append(sp.sum((abs(psi_sum - det_mod[i]) ** 2.).data[0]) / sp.sum(det_mod[i].data[0] ** 2.))
        return indvdl_mse

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

        self.slow_db_queue['fresnel_number'] = (nNF,)
        self.slow_db_queue['oversampling'] = (nOS,)
        self.slow_db_queue['dx_s'] = (del_x_s(l, z, x),)
        self.slow_db_queue['sample_fov'] = (del_x_s(l, z, x)*pix,)
        self.slow_db_queue['numerical_aperture'] = (NA,)
        self.slow_db_queue['axial_resolution'] = (axial_res,)

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
                shutil.copy(CXP.io.code_dir+'/CXParams.py', _py_dir)
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
                shutil.copy(CXP.io.code_dir+'/CXParams.py', _py_dir)
                shutil.copy(CXP.io.code_dir+'/CXParams.py',
                            _cur_sequence_dir+'/CXParams_sequence{}.py'.format(CXP.reconstruction.sequence))
            except IOError:
                CXP.log.error('Was unable to save a copy of CXParams.py to {}'.format(_py_dir))

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
        CXP.log.info('Getting ptycho position mesh.')
        
        if CXP.measurement.ptycho_scan_mesh == 'generate':
            if CXP.measurement.ptycho_scan_type == 'cartesian':
                x2 = 0.5*(CXP.measurement.cartesian_scan_dims[0]-1)
                y2 = 0.5*(CXP.measurement.cartesian_scan_dims[1]-1)
                tmp = map(lambda a: CXP.measurement.cartesian_step_size*a, np.mgrid[-x2:x2+1, -y2:y2+1])
                self.positions.data = [tmp[0].flatten(), tmp[1].flatten()]
                if CXP.reconstruction.flip_mesh_lr:
                    self.log.info('Flip ptycho mesh left-right')
                    self.positions.data[0] = self.data[0][::-1]
                if CXP.reconstruction.flip_mesh_ud:
                    self.log.info('Flip ptycho mesh up-down')
                    self.positions.data[1] = self.data[1][::-1]
                if CXP.reconstruction.flip_fast_axis:
                    self.log.info('Flip ptycho mesh fast axis')
                    tmp0, tmp1 = self.data[0], self.data[1]
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
            probe = abs(gaussian*probe)* exp(complex(0.,np.pi)*ph_func/ph_func.max())

            self.probe.modes = [CXData(data=[probe/(i+1)]) for i in range(CXP.reconstruction.probe_modes)]
            
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