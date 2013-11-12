###############
Getting Started 
###############

CXPhasing is a python module for performing ptychographic phase retrieval. It is based on the `ePIE algorithm <http://dx.doi.org/10.1016/j.ultramic.2009.05.012>`_ and implements position correction and state-mixture recovery.
CXPhasing uses a MySQL database to store all the parameters for each reconstruction attempt.

Installation
-------------
CXPhasing can be installed:

	#. from the Python Package Index via::

		> pip install cxphasing

	#. from github::

		> git clone https://github.com/djvine/cxphasing.git
		> cd cxphasing
		> python setup.py install

If the installation worked you will be able to import cxphasing from a python terminal::

	> import cxphasing

Specifying Reconstruction Parameters
------------------------------------
All of the parameters are stored in the CXParams file.
