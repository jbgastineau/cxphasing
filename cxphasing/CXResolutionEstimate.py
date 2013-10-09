"""
.. module:: CXResolutionEstimate.py
   :platform: Unix
   :synopsis: A class for predicting the resolution of a ptychography measurement.

.. moduleauthor:: David Vine <djvine@gmail.com>


"""
import requests
import pdb
import scipy as sp
import numpy as np
import scipy.fftpack as spf
from pylab import *

class HenkeRefractiveIndex(object):
	""".. class:: HenkeRefractiveIndex()

		A class for interacting with henke.lbl.gov to get the complex
		refractive index of a material at a given energy.

		:attr float energy: X-ray energy [keV].
		:attr str formula: Chemical formula.
		:attr float density: Material density [gm/cm^3].
		:attr str url: POST form url.
		:attr dict form_data: Stores POST form data.



	"""

	def __init__(self, energy='1', formula='Si3N4', density='-1'):
		self.url = 'http://henke.lbl.gov/cgi-bin/getdb.pl'

		self.form_data = {'Density': density,
 					 'Formula': formula,
 					 'Max': energy,
 					 'Min': energy,
 					 'Npts': '10',
 					 'Output': 'Text File',
 					 'submit': 'submit'
 					}

		self.get_response()


	def get_response(self):
		response = requests.post(self.url, data=self.form_data)
		url = response.content('HREF="')[1].split('"')[0]
		result = requests.get('http://henke.lbl.gov'+url).split('\n')[2].split(' ')
		self.delta = result[4]
		self.beta = result[6]


class Detector(object):

	""".. class:: Detector([pilatus100k, pilatus1M, medipix])

		A class for describing X-ray area detectors.

		:attr int xpix: Number of pixels in x direction.
		:attr int ypix: Number of pixels in y direction.
		:attr float pix_size: Pixel size [m].
		:attr int dr: Dynamic range.
		:attr tuple nchips: Chip arrangement.
		:attr int dead_space: Dead space between chips [units of detector pixels].

	"""
	def __init__(self, det):

		if det=='pilatus100k':
			d = {
				'xpix':187,
				'ypix':485,
				'pix_size':172e-6,
				'dr': 2**20,
				'nchips': (1,1),
				'dead_space': 0
				}
		elif det=='pilatus1M':
			d = {
				'xpix':187,
				'ypix':485,
				'pix_size':172e-6,
				'dr': 2**20,
				'nchips': (2,3),
				'dead_space': 4
				}
		elif det=='medipix':
			d = {
				'xpix':256,
				'ypix':256,
				'pix_size':55e-6,
				'dr': 11800,
				'nchips': (1,1),
				'dead_space': 0
				}

		for k, v in d.iteritems():
			setattr(self, k, v)

class TransmissionFunction(object):
	""".. class:: TransmissionFunction()

		Class for calculating the transmission function.

		:attr np.ndarray mag: absolute value of complex refractive index.
		:attr np.ndarray pha: phase of complex refractive index.
		:attr np.ndarray thickness: the thickness function [m].
		:default thickness: lena.
		:attr float max_thickness: Thickness function scaled to this max thickness [micron].
		:default max_thickness: 1.0.
		:attr str material: the material the sample is composed of: [gold, protein].
		:attr float density: material density.
		:attr np.ndarray T: complex transmission function.

	"""
	def __init__(self, thickness=sp.misc.lena(), max_thickness=1.0, energy=1.0, **kwargs):

		thickness -= thickness.min()
		thickness *= max_thickness*1e-6/thickness.max()

		l = 1.24e-9/energy

		ref_ind = HenkeRefractiveIndex(energy, **kwargs)

		self.T = exp((2.0*math.pi/l)*complex(ref_ind.beta, ref_ind.delta)*thickness)

		for k, v in kwargs.iteritems():
			setattr(self, k, v)

	def __mul__(self, a, b):
		return TransmissionFunction(T=a.T*b.T)

	@staticmethod
	def fft(a):
		return TransmissionFunction(T=spf.fftshift(spf.fft2(a.T)))

	def show(self):
		pylab.matshow(sp.abs(self.T)**2.0)

class Params(object):

	""".. class:: Params()

		A class for storing parameters for the simulation.

		:attr float energy: X-ray energy [keV].
		:attr float z: Sample-detector distance [m].
		:attr Detector det: Type of area detector. Choices ['pilatus100k', 'pilatus1M', 'medipix'].
		:attr float zp_diameter: Zone plate diameter [micron].
		:attr float zp_finest_zone: Zone plate finest zone [micron].
		:attr float beamstop_radius: Beamtop radius [micron].
		:attr float beamstop thickness: Beamstop thickness [micron].
		:attr str beamstop_material: Beamstop material.
		:attr TransmissionFunction beamstop: Beamstop transmission function.
		:attr TransmissionFunction sample: Sample transmission function.

	"""
	def __init__(self):

		self.energy = 10.0
		self.l = 1.24e-9/self.energy
		self.z = 1.0
		self.det = Detector('pilatus100k')

		self.zp_diameter = 160.0
		self.zp_finest_zone = 80.0
		self.zp_focal_length = self.zp_diameter*1e-6*self.zp_finest_zone*1e-9/self.l

		self.beamstop_size = 100.0
		self.beamstop_thickness = 100.0
		self.beamstop_material = 'Si'

		self.dx_s = self.l*self.z/(min(self.det.xpix, self.det.ypix)*self.det.pix_size)
		self.dx_zp = self.zp_focal_length*self.det.pix_size/self.z

class IncidentIllumination(object):
	""".. class:: IncidentIllumination(zp_radius, dx_zp)

		Calculate the incident illumination in the sample plane.

		:attr float zp_radius: Zone plate radius [micron].
		:attr float dx_zp: Array physical pixel size [m].
		:attr numpy.ndarray T: Array containing complex wavefield describing incident illumination.
	"""

	def __init__(self, zp_radius, dx_zp):

		zp_radius_in_pixels = sp.ceil(zp_radius*1e-6/dx_zp)
		self.T = spf.fftshift(spf.fft2(sp.where(sp.hypot(*sp.ogrid[-1024:1024, -1024:1024])<zp_radius_in_pixels, 1., 0.)))


def recommend_beamstop(q_dependence=-3.5, energy=10.0, det=Detector('pilatus100k'), z_or_dx={'dx': 10e-9}):

	""".. func:: recommend_beamstop(q_dependence, detector, z_or_dx)

		:param float q_dependence: Intensity vs q scaling in far field.
		:param float energy: Incident X-ray energy.
		:param Detector det: Detector to be used for calculation.
		:param dict z_or_dx: Choose the optimal beamstop when (i) the detector is placed at z or, (ii) the desired resolution is dx.

	"""

	det_npix = min(det.xpix, det.ypix)
	det_width = det.pix_size*det_npix/2
	l = 1.24e-9/energy

	if 'z' in z_or_dx.keys():
		z = z_or_dx['z']
		dx = l*z/det_width
	else:
		dx = z_or_dx['dx']
		z = dx*det_width/(2*l)

	det_domain_x = sp.arange(det_npix)*det.pix_size
	det_domain_q = 4*math.pi*det_domain_x/(l*z)
	
	intensity = lambda q: (1+q**2.0)**-2.0

	full_dynamic_range = log10(intensity(det_domain_q[-1]/det_domain_q[0]))

	detector_dynamic_range = log10(det.dr)

	required_dynamic_range = full_dynamic_range-detector_dynamic_range

	# Is a beamstop required?
	if required_dynamic_range>0: 
		# Yes
		pass


def main():
	pdb.set_trace()
	p = Params()

	i0 = IncidentIllumination(p.zp_diameter/2.0, p.dx_zp)

	sample = TransmissionFunction(energy=p.energy)
	beamstop = TransmissionFunction(thickness=sp.ones((3,3)), max_thickness=p.beamstop_thickness, 
			material=p.beamstop_material, energy=p.energy)

	exit_wave = i0*sample

	det_wave = TransmissionFunction.fft(exit_wave) * beamstop

	det_wave.show()

	# Do photon scaling

if __name__=='__main__': main()	

