from nose.tools import *
from cxphasing.CXData2 import CXData, CXModal
import numpy as np
import scipy as sp 
lena = sp.misc.lena()

def test_init():

	psi = CXModal(modes=[CXData(data=lena) for i in range(3)])
	assert True

def test_get_item():
	psi = CXModal(modes=[CXData(data=[lena for j in range(4)]) for i in range(3)])
	data = CXData(data=[lena for j in range(4)])
	assert(isinstance(psi[0], CXData))
	assert(np.array_equal(psi[0].data[0], lena))

def test_set_item():
	psi = CXModal(modes=[CXData(data=[lena for j in range(4)]) for i in range(3)])
	data = CXData(data=[lena for j in range(4)])

	psi[0] = CXData(data=[sp.zeros((128,128)) for j in range(4)])
	for i in range(4):
		assert(np.array_equal(psi[0].data[i], sp.zeros((128,128))))

def test_add():
	psi = CXModal(modes=[CXData(data=[lena for j in range(4)]) for i in range(3)])
	data = CXData(data=[lena for j in range(4)])
	psi2 = psi+psi
	psidata = psi+data
	psi3 = psi+3
	for mode in range(3):
		for i in range(4):
			assert(np.array_equal(psi2[mode].data[i], 2*lena))
			assert(np.array_equal(psidata[mode].data[i], 2*lena))
			assert(np.array_equal(psi3[mode].data[i], lena+3))

def test_iadd():
	psi = CXModal(modes=[CXData(data=[lena.copy() for j in range(4)]) for i in range(3)])
	psi1 = CXModal(modes=[CXData(data=[lena.copy() for j in range(4)]) for i in range(3)])
	psi2 = CXModal(modes=[CXData(data=[lena.copy() for j in range(4)]) for i in range(3)])
	psi3 = CXModal(modes=[CXData(data=[lena.copy() for j in range(4)]) for i in range(3)])
	data = CXData(data=[lena for j in range(4)])


	psi1 += psi
	psi2 += data
	psi3 += 3
	for mode in range(3):
		for i in range(4):
			assert(np.array_equal(psi1[mode].data[i], 2*lena))
			assert(np.array_equal(psi2[mode].data[i], 2*lena))
			assert(np.array_equal(psi3[mode].data[i], lena+3))
