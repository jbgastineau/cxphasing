#!/usr/bin/python
import numpy as np
import vine_utils as vu
import CXParams as CXP
from CXData import CXData
from cxparams import CXParams as CXP
from CXData import fft2, ifft2, angle, exp, fftshift, conj, abs
from CXData import log as nlog
import time
import pdb
import scipy as sp
from numpy.random import uniform
import pylab
from CXDb import SimpleDB
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import multiprocessing as mp
import multiprocess
import itertools
from colorsys import hls_to_rgb 
v_hls_to_rgb = np.vectorize(hls_to_rgb)


#constant
machine_precision = 2e-14

log = None

#np.random.seed(86585485)


def worker(func):

    def worker2(self=None, *args, **kwargs):
        name = mp.current_process().name
        cnt = 0
        jobs, results = args[0], args[1]
        while True:
            i, view, psi, det_mod = jobs.get()

            if i == None:  # Deal with Poison Pill
                print '{}: Exiting. {:d} jobs completed.'.format(name, cnt)
                jobs.task_done()
                break

            res = func(self, i, view, psi, det_mod)
            cnt += 1
            jobs.task_done()
            results.put(res)
        return worker2
    return worker2


def object_worker(func):

    def object_worker2(self=None, *args, **kwargs):
        name = mp.current_process().name
        cnt = 0
        jobs, results = args[0], args[1]
        while True:
            psi, probe, positions = jobs.get()

            if psi == None:  # Deal with Poison Pill
                print '{}: Exiting. {:d} jobs completed.'.format(name, cnt)
                jobs.task_done()
                break

            res = func(self, psi, probe, positions)
            cnt += 1
            jobs.task_done()
            results.put(res)
        return object_worker2
    return object_worker2


def split_seq(seq, num_pieces):
    """ split a list into pieces passed as param """
    start = 0
    for i in xrange(num_pieces):
        stop = start + len(seq[i::num_pieces])
        yield seq[start:stop]
        start = stop


class CXPhasing(CXData):

    def __init__(self, *args):
        if not hasattr(self, 'log'):
            self.log = CXP.log

        if args:
            self.args = args

        else:
            self.setup_dir_tree()

            self.choose_actions()

            if CXP.measurement.simulate_data:
                self.simulate_data()
                self.det_mod = CXData()
                self.probe_det_mod = CXData()
                self.det_mod.filename = CXData.processed_filename_string.format('det_mod')
                self.det_mod.load()
                self.probe_det_mod.filename = CXData.processed_filename_string.format('probe_det_mod')
                self.probe_det_mod.load()
            else:
                self.add_object('det_mod')
                for ob in ['probe_det_mod']:
                    if CXP.io.whitefield_filename:
                        self.add_object(ob)

            # Calculate STXM image if this is a ptycho scan
            if len(self.det_mod.data) > 1:
                self.det_mod.calc_stxm_image()

            if CXP.actions.process_dpc:
                self.process_dpc()

            self.total_its = 0
            obs = ['object', 'probe', 'positions', 'psi', 'probe_mod_squared',
                   'probe_mod_squared_map', 'ob_mod_squared_map']
            for ob in obs:
                self.add_object(ob)

            # Probe position correction
            self.annealing_schedule = {'linear': lambda x: 1 if x ==0 else np.max([0.05,
                            1. - np.double(x) / CXP.reconstruction.ppc_length]),
                                       'classic': lambda x: np.log(1+np.double(x))**-1.,
                                       'fast': lambda x: (1+np.double(x))**-1.
                                       }[CXP.reconstruction.ppc_annealing_schedule]
            self.ppc = False

            # MySQL DB Integration
            if CXP.hasmysql:
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

            self.p = self.det_mod.data[0].shape[0]
            CXP.p = self.p
            self.p2 = self.p / 2
            # Calculate object array size based on ptycho mesh size
            self.ob_p = self.calc_ob_shape()

            self.algorithms = {
                                'dm': self.dm,
                                'er': self.er
                            }

            self.algorithm = self.algorithms['dm']

            if CXP.machine.n_processes < 0:
                CXP.machine.n_processes = mp.cpu_count()

            self.dm_repr = 'Difference Map\n\tPtychography iteration:{:10d}\n\tPtychography position:{:10d} [{:3.0f}%]'
            self.er_repr = 'Error reduction\n\tPtychography iteration:{:10d}\n\tPtychography position:{:10d} [{:3.0f}%]'
            self.progress_repr = 'Current iteration: {:d}\tPosition: {:d}'

            self.init_figure()
            self.log_reconstruction_parameters()

    def __call__(self):
        return getattr(self, self.args[0])(*self.args[1:])

    def add_object(self, ob):
        try:
            setattr(self, ob, CXData.__all__[ob])
        except KeyError:
            setattr(self, ob, CXData(itype=ob))

    def init_logging(self):
        global log

        log = self.log

    def reg_filt(self, x):
        return CXData(data=x.data[0].max()*((1 - (x.data[0] / x.data[0].max())) ** 10.))

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

    def M(self, psi, det_mod):
        return ifft2(det_mod * exp(complex(0., 1.) * angle(fft2(psi))))

    def error(self, psi, det_mod):
        return sp.sum((abs(abs(fft2(psi)) - det_mod) ** 2.).data[0]) / sp.sum(det_mod.data[0] ** 2.)

    def update_object_single_position(self, i):
        """
        Update the object from a single ptycho position.

        Pseudo code
        object += probe*psi/(|probe|**2+|probe|**-4)
        or
        object += probe*psi/(|probe|**2+alpha*max(|probe|**2))
            where alpha is typically 1-10%.

        """
        if not (self.total_its + i):
            self.probe_mod_squared = abs(self.probe) ** 2.

        d1, d2 = self.positions.data
        id1, id2 = d1//1, d2//1

        self.object[id1[i] - self.p2:id1[i] + self.p2, id2[i] - self.p2:id2[i] + self.p2] += \
                            self.shift(conj(self.probe) * self.psi[i] / (self.probe_mod_squared +
                             self.reg_filt(self.probe_mod_squared)), d1[i]%1, d2[i]%1)

        if sp.mod(i, len(self.positions.data[0]) / 10) == 0:
            self.update_figure(i)

    def update_object(self):
        then = time.time()
        if ((self.itnum == 1) or
            (self.itnum > CXP.reconstruction.begin_updating_probe) or
            (self.itnum > CXP.reconstruction.ppc_begin)):
            self.update_probe_map()

        multip = multiprocess.multiprocess(self.update_object_worker)

        for element in list(split_seq(range(self.positions.total),
                CXP.machine.n_processes)):

            multip.add_job((CXData(data=self.psi.data[element[0]:element[-1]+1]), self.probe,
                (self.positions.data[0][element[0]:element[-1]+1],
                 self.positions.data[1][element[0]:element[-1]+1]), self.object))

        results = multip.close_out()

        for result in results:
            self.object+=result

        self.object /= (self.probe_mod_squared_map + self.reg_filt(self.probe_mod_squared_map))
        self.log.info('{:3.2f} seconds elapsed updating object.'.format(time.time() - then))
        self.fast_db_queue['object_update_elapsed_secs'] = (self.itnum, time.time() - then)
        self.object.save()

        self.update_figure()

    @staticmethod
    @multiprocess.worker
    def update_object_worker(args):
        psi, probe, positions, orig_obj = args
        d1, d2 = positions
        p2 = probe.data[0].shape[0]/2
        id1, id2 = d1//1, d2//1
        obj = CXData(data=sp.zeros(orig_obj.data[0].shape, np.complex))
        for i in range(len(psi)):
            obj[id1[i] - p2: id1[i] + p2, id2[i] - p2: id2[i] + p2] += CXData.shift(conj(probe) * psi[i],
                                                                            d1[i]%1, d2[i]%1)
        return obj

    def reconstitute_object(self, old_pos, new_pos):
        """
        The object has "memory" of previous positions when updating it only from psi.
        This reconsititution resets the object at the current positions
        """

        od1, od2 = old_pos
        nd1, nd2 = new_pos

        multip = multiprocess.multiprocess(self.update_object_worker)
        psi = CXData(data=[])
        for i in range(self.positions.total):
            psi.data.append((self.probe * self.object[od1[i] - self.p2:od1[i] + self.p2,
                                                      od2[i] - self.p2:od2[i] + self.p2]).data[0])

        for element in list(split_seq(range(self.positions.total),
                CXP.machine.n_processes)):

            multip.add_job((CXData(data=psi.data[element[0]:element[-1]+1]), self.probe,
                (nd1[element[0]:element[-1]+1],
                 nd2[element[0]:element[-1]+1]),
                self.object))

        results = multip.close_out()

        self.update_probe_map()
        self.object.data[0] = sp.zeros(self.object.data[0].shape, np.complex)
        for result in results:
            self.object+=result

        self.object /= (self.probe_mod_squared_map + self.reg_filt(self.probe_mod_squared_map))

    def update_probe(self):
        then = time.time()
        self.update_object_map()
        p2 = self.p2

        d1, d2 = self.positions.data
        for i in range(self.positions.total):
            self.probe += conj(self.object[d1[i] - p2:d1[i] + p2, d2[i] - p2:d2[i] + p2]) * self.psi[i]

        self.probe /= (self.ob_mod_squared_map + self.reg_filt(self.ob_mod_squared_map))
        try:
            self.probe[0] = self.M(self.probe, self.probe_det_mod)
        except AttributeError:
            self.probe.normalise()

        self.probe_mod_squared = abs(self.probe) ** 2.
        self.log.info('{:3.2f} seconds elapsed updating probe.'.format(time.time() - then))
        self.fast_db_queue['probe_update_elapsed_secs'] = (self.itnum, time.time() - then)
        self.probe.save()

    def initial_update_state_vector(self):

        d1, d2 = self.positions.data
        for i in xrange(self.positions.total):
            if i % np.floor(self.positions.total / 10) == 0 and CXP.reconstruction.verbose:
                self.log.info(self.dm_repr.format(self.itnum, i, 100. * float(i + 1) / self.positions.total))
            view = self.probe * self.object[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]

            if not (self.itnum + i):
                self.psi[i] = self.M(self.probe, self.det_mod[i])
            else:
                self.psi[i] += self.M(2. * view - self.psi[i], self.det_mod[i]) - view

            self.update_object_single_position(i)

        for i in xrange(self.positions.total):
            view = self.probe * self.object[d1[i] - self.p2:d1[i] + self.p2, d2[i] - self.p2:d2[i] + self.p2]
            self.psi[i] = self.M(self.probe, self.det_mod[i])

    def update_state_vector(self):

        if not self.total_its:
            self.initial_update_state_vector()

        else:
            d1, d2 = self.positions.data

            jobs = mp.JoinableQueue()
            results = mp.Queue()

            p = [mp.Process(target=self.algorithm, args=(jobs, results))
                        for i in range(CXP.machine.n_processes)]

            for process in p:
                process.start()

            then = time.time()
            ppc_dict = {}
            ppc_error = sp.ones(self.positions.total)

            for i in xrange(self.positions.total):
                jobs.put(([i, None],  [self.probe, self.object, d1[i], d2[i], self.p2], self.psi[i], self.det_mod[i]))
                ppc_dict['{:d}'.format(i)] = [0, 0]
                if self.ppc:
                    old_pos = self.positions.data
                    if i == 0:
                        self.log.info('Annealing schedule: {}. Value {:3.2f}'.format(CXP.reconstruction.ppc_annealing_schedule,
                                            self.annealing_schedule(self.itnum - CXP.reconstruction.ppc_begin)))
                    for j in range(1, 1 + CXP.reconstruction.ppc_trial_positions):
                        cx, cy = self.pos_correction_transform(i, self.itnum - CXP.reconstruction.ppc_begin)
                        ppc_dict['{:d}_{:d}'.format(i, j)] = [cx, cy]
                        jobs.put(([i, j], [self.probe, self.object, cx, cy, self.p2], self.psi[i], self.det_mod[i]))

            # Add Poison Pill
            for i in range(CXP.machine.n_processes):
                jobs.put((None, None, None, None))

            self.log.info('{:3.2f} seconds elapsed dividing jobs between processes.'.format(time.time() - then))
            cnt = 0
            while True:
                if not results.empty():
                    ij_list, psi, error = results.get()
                    i, j = ij_list
                    if self.ppc:
                        if error < ppc_error[i]:
                            ppc_error[i] = error
                            self.psi[i] = psi
                            if j:
                                self.positions.data[0][i] = ppc_dict['{:d}_{:d}'.format(i, j)][0]
                                self.positions.data[1][i] = ppc_dict['{:d}_{:d}'.format(i, j)][1]
                    else:
                        self.psi[i] = psi
                    cnt += 1
                if cnt == len(ppc_dict):
                    break

            jobs.join()

            jobs.close()
            results.close()

            for process in p:
                process.join()

            if self.ppc:
                self.reconstitute_object(old_pos, self.positions.data)
            #else:
            #    self.reconstitute_object(self.positions.data, self.positions.data)

    @worker
    def dm(self, ij_list, view_data, psi, det_mod):
        probe, ob, d1, d2, p2 = view_data
        i, j = ij_list
        view = probe * ob[d1 - p2:d1 + p2, d2 - p2:d2 + p2]
        error = self.error(view, det_mod)

        if i % np.floor(self.positions.total / 10) == 0 and CXP.reconstruction.verbose and not j:
            self.log.info(self.dm_repr.format(self.itnum, i, 100. * float(i + 1) / self.positions.total))

        psi += self.M(2. * view - psi, det_mod) - view

        return (ij_list, psi, error)

    @worker
    def er(self, ij_list, view_data, psi, det_mod):
        probe, ob, d1, d2, p2 = view_data
        i, j = ij_list
        view = probe * ob[d1 - p2:d1 + p2, d2 - p2:d2 + p2]
        error = self.error(view, det_mod)

        if i % np.floor(self.positions.total / 10) == 0 and CXP.reconstruction.verbose and not j:
            self.log.info(self.er_repr.format(self.itnum, i, 100. * float(i + 1) / self.positions.total))

        return (ij_list, self.M(view, det_mod), error)

    def update_probe_map(self):
        p2 = self.p2
        probe_mod_squared = abs(self.probe) ** 2.
        self.probe_mod_squared_map = CXData(data=sp.zeros((self.ob_p, self.ob_p)))
        d1, d2 = self.positions.data
        id1, id2 = d1//1, d2//1
        for i in range(self.positions.total):
            self.probe_mod_squared_map[id1[i] - p2:id1[i] + p2, id2[i] - p2:id2[i] + p2] += self.shift(
                                    probe_mod_squared, d1[i]%1., d2[i]%1.)

    def update_object_map(self):
        p2 = self.p2
        ob_mod_squared = abs(self.object) ** 2.
        self.ob_mod_squared_map = CXData(data=sp.zeros((self.p, self.p)))
        d1, d2 = self.positions.data
        for i in range(self.positions.total):
            self.ob_mod_squared_map += ob_mod_squared[d1[i] - p2:d1[i] + p2, d2[i] - p2:d2[i] + p2]

    def select_algorithm(self):
        try:
            self.ER_count
        except AttributeError:
            self.ER_count = 0

        if self.algorithm == self.algorithms['er']:
            self.algorithm = self.algorithms['dm']

        if self.ER_count > CXP.reconstruction.ER_n:
            self.algorithm = self.algorithms['er']
            self.ER_count = 0

        self.ER_count += 1
        self.fast_db_queue['algorithm'] = (self.itnum, self.algorithm.func_name)

    def run(self, its=None):
        if not its:
            its = CXP.reconstruction.ptycho_its
        if CXP.hasmysql:
            self.update_slow_table()
        beginning = time.time()

        for self.itnum in xrange(its):
            then = time.time()

            self.select_algorithm()

            self.update_state_vector()

            if self.itnum > 0 and self.itnum < CXP.reconstruction.begin_updating_probe:
                self.update_object()
            elif self.itnum > CXP.reconstruction.begin_updating_probe:
                for m in range(CXP.reconstruction.probe_object_its):
                    self.update_object()
                    self.update_probe()
            if self.itnum >= CXP.reconstruction.ppc_begin and \
                CXP.reconstruction.probe_position_correction:
                self.ppc = True

            now = time.time()
            self.fast_db_queue['iter_time'] = (self.itnum, now - then)
            self.fast_db_queue['iter_time_pptpxit'] = (self.itnum, 1e6*(now - then) / (self.positions.total * self.p**2 * (self.itnum + 1)))
            self.log.info('{:2.2f} seconds elapsed during iteration {:d} [{:1.2e} sec/pt/pix/it]'.format(now - then, self.itnum + 1,
                            (now-then)/(self.positions.total * self.p**2 * (self.itnum + 1))))
            self.log.info('{:5.2f} seconds have elapsed in {:d} iterations [{:2.2f} sec/it]'.format(now-beginning, self.itnum + 1, (now-beginning)/(self.total_its + 1)))
            self.calc_mse()
            self.total_its += 1
            if CXP.hasmysql:
                self.update_fast_table()
            if self.itnum > 0:
                self.update_figure(self.itnum)

    def init_figure(self):
        pylab.ion()
        self.f1=pylab.figure(1, figsize=(12, 10))
        try:
            itnum = self.itnum
        except AttributeError:
            itnum = 0
        try:
            mse = self.av_mse
        except AttributeError:
            mse = -1.0
        pylab.suptitle('Sequence: {:d}, Iteration: {:d}, MSE: {:3.2f}%'.format(CXP.reconstruction.sequence, itnum, 100*mse))

    def update_figure(self, i=None):
        cur_cmap = cm.RdGy_r
        if not i:
            i = self.total_its
        self.f1.clf()
        self.init_figure()
        wh = sp.where(abs(self.object.data[0]) > 0.1 * (abs(self.object.data[0]).max()))
        try:
            x1, x2 = min(wh[0]), max(wh[0])
            y1, y2 = min(wh[1]), max(wh[1])
        except (ValueError, IndexError):
            pdb.set_trace()
            x1, x2 = 0, self.ob_p
            y1, y2 = 0, self.ob_p
        s1 = pylab.subplot(221)
        s1_im = s1.imshow(abs(self.object).data[0][x1:x2, y1:y2], cmap=cm.Greys_r)
        pylab.colorbar(s1_im)
        s2 = pylab.subplot(222)
        h = ((angle(self.object).data[0][x1:x2, y1:y2] + np.pi) / (2*np.pi)) % 1.0
        s = np.ones_like(h)
        l = abs(self.object).data[0][x1:x2, y1:y2]
        l-=l.min()
        l/=l.max()
        s2_im = s2.imshow(np.dstack(v_hls_to_rgb(h,l,s)))
        pylab.colorbar(s2_im)
        s3 = pylab.subplot(223)
        s3_im = s3.imshow(abs(self.probe).data[0], cmap=cur_cmap)
        pylab.colorbar(s3_im)
        if self.ppc:
            s4 = self.f1.add_subplot(224)
            s4_im = s4.scatter(self.positions.data[0], self.positions.data[1], s=10,
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
                self.log.info('RMS position deviation from correct: [x:{:3.2f},y:{:3.2f}] pixels'.format(
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
                    s4.plot(x, y, 'g-')
                self.log.info('RMS position deviation from initial: [x:{:3.2f},y:{:3.2f}] pixels'.format(
                            sp.sqrt(sp.mean((self.positions.data[0] - self.positions.initial[0])**2.)),
                            sp.sqrt(sp.mean((self.positions.data[1] - self.positions.initial[1])**2.))))
            s4.legend(prop={'size': 6})
            s4.set_title('Position Correction')
            s4.set_aspect('equal')
            extent = s4.get_window_extent().transformed(self.f1.dpi_scale_trans.inverted())
            pylab.savefig(self._cur_sequence_dir + '/ppc_{:d}.png'.format(self.total_its), bbox_inches=extent.expanded(1.2, 1.2), dpi=100)
            s4.set_aspect('auto')
        else:
            s4 = pylab.subplot(224)
            s4_im = s4.imshow(nlog(fftshift(self.det_mod[i//len(self.det_mod.data)])).data[0], cmap=cur_cmap)
            pylab.colorbar(s4_im)
        pylab.draw()
        pylab.savefig(self._cur_sequence_dir + '/recon_{:d}.png'.format(self.total_its), dpi=60)

    def simulate_data(self):
        self.log.info('Simulating diffraction patterns.')
        sample = CXData()
        det_mod = CXData()
        probe_det_mod = CXData()
        sample.load(CXP.io.simulation_sample_filename[0])
        if len(CXP.io.simulation_sample_filename)>1:
            ph = CXData()
            ph.load(CXP.io.simulation_sample_filename[1])
            ph.normalise(val=np.pi/3)
            sample.data[0] = sample.data[0]*exp(complex(0., 1.)*ph.data[0])
        det_mod.simulate_diffraction_patterns(sample)
        probe_det_mod.init_data('probe')
        probe_det_mod.data = [abs(fft2(probe_det_mod.data[0]))]

        det_mod.save(CXData.processed_filename_string.format('det_mod'))
        probe_det_mod.save(CXData.processed_filename_string.format('probe_det_mod'))

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
        self.log.info('MySQL Reconstruction ID: {}'.format(self.recon_id))

    def update_slow_table(self):

        for element in CXP.param_store._priority:
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
        self.log.info('{:3.2f} seconds elapsed entering {:d} values into slow db [{:3.2f} msec/entry]'.format(now-then,
                        cnt, 1e3*(now - then) / cnt))

    def update_fast_table(self):

        if not self.t_fast_params.check_columns(self.fast_db_queue.keys()):
            for key, (itnum, value) in self.fast_db_queue.iteritems():
                if not self.t_fast_params.check_columns([key]):
                    self.log.warning('MYSQL: Adding column {} to fast_params.'.format(key))
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
        self.log.info('{:3.2f} seconds elapsed entering {:d} values into fast db [{:3.2f} msec/entry]'.format(now-then,
                        cnt, 1e3 * (now - then) / cnt))

    def calc_mse(self):
        then = time.time()

        multip = multiprocess.multiprocess(self.mse_worker)

        d1, d2 = self.positions.data

        for i_range in list(split_seq(range(self.positions.total),
                CXP.machine.n_processes)):
            multip.add_job((i_range, d1, d2, self.probe, self.object, self.det_mod))

        results = multip.close_out()

        self.av_mse = sp.mean(list(itertools.chain(*results)))

        self.log.info('Mean square error: {:3.2f}%'.format(100 * self.av_mse))
        self.fast_db_queue['error'] = (self.itnum, self.av_mse)
        now = time.time()
        self.log.info('Calculating MSE took {:3.2f}sec [{:3.2f}msec/position]'.format(now - then,
                       1e3*(now - then) / self.positions.total))

    @staticmethod
    @multiprocess.worker
    def mse_worker(args):
        i_range, d1, d2, probe, obj, det_mod = args
        p2 = probe.data[0].shape[0]/2
        indvdl_mse = []
        for i in i_range:
            view = probe * obj[d1[i] - p2:d1[i] + p2, d2[i] - p2:d2[i] + p2]
            indvdl_mse.append(sp.sum((abs(abs(fft2(view)) - det_mod[i]) ** 2.).data[0]) / sp.sum(det_mod[i].data[0] ** 2.))
        return indvdl_mse

    def log_reconstruction_parameters(self):
        """
        h - object size\nz - sam-det dist\npix - # of pix\ndel_x_d - pixel size
        """
        dx_d = CXP.experiment.dx_d
        x = (CXP.p/2.)*dx_d
        l = vu.energy_to_wavelength(CXP.experiment.energy)
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
        self.log.info('Fresnel number: {:2.2e}'.format(nNF))
        self.log.info('Oversampling: {:3.2f}'.format(nOS))
        self.log.info('Detector pixel size: {:3.2f} [micron]'.format(1e6*dx_d))
        self.log.info('Detector width: {:3.2f} [mm]'.format(1e3*pix*dx_d))
        self.log.info('Sample pixel size: {:3.2f} [nm]'.format(1e9*del_x_s(l, z, x)))
        self.log.info('Sample FOV: {:3.2f} [micron]'.format(1e6*del_x_s(l, z, x)*pix))
        self.log.info('Numerical aperture: {:3.2f}'.format(NA))
        self.log.info('Axial resolution: {:3.2f} [micron]'.format(1e6*axial_res))
        self.log.info('Lateral resolution: {:3.2f} [nm]'.format(1e9*lateral_res))

        self.slow_db_queue['fresnel_number'] = (nNF,)
        self.slow_db_queue['oversampling'] = (nOS,)
        self.slow_db_queue['dx_s'] = (del_x_s(l, z, x),)
        self.slow_db_queue['sample_fov'] = (del_x_s(l, z, x)*pix,)
        self.slow_db_queue['numerical_aperture'] = (NA,)
        self.slow_db_queue['axial_resolution'] = (axial_res,)

        #self.slow_db_queue['rms_rounding_error_x'] = (CXP.rms_rounding_error[0],)
        #self.slow_db_queue['rms_rounding_error_y'] = (CXP.rms_rounding_error[1],)

if __name__ == '__main__':
    CXPh = CXPhasing()
    CXPh.run()
