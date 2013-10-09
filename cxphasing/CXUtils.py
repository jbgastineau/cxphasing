from colorsys import hls_to_rgb 
import numpy as np
import scipy as sp
import scipy.fftpack as spf
import scipy.constants as spc
import multiprocessing as mp
import multiprocess
v_hls_to_rgb = np.vectorize(hls_to_rgb)

def energy_to_wavelength(energy):
    """
    Converts an energy in keV to wavelength in metres.
    """
    wl = spc.physical_constants['Planck constant'][0]
    wl *= spc.speed_of_light
    wl /= (energy*1e3*spc.elementary_charge)
    
    return wl

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

def gauss_smooth(arr, w):
    """
    Smooths an input array by w pixels
    """
    alpha = w/2.3548
    pix = arr.shape[0]
    ker = sp.exp(-(alpha**-2.)*sp.hypot(*sp.ogrid[-pix/2:pix/2,-pix/2:pix/2])**2.)
    return sp.real(spf.fftshift(spf.ifft2( spf.fft2(arr)*spf.fft2(ker) )))

def tukeywin(window_length, alpha=0.5):
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
        CXP.log.error('Must define a filename to create a gif')
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
