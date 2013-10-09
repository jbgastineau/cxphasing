# Python code to generate the Cartesian coordinates of a "round scan" as described in 
# Dierolf et al. New J. Phys. 12, 035017 (2010). This is to prevent known artifacts
# in Ptychographic Coherent Diffractive Imaging which arise from scans on Cartesian
# grids. 
# 
# Written by K. Giewekemeyer, European XFEL (2012)
# Modified DJVine June 2012

import numpy as np
import scipy as sp
from math import pi

def round_roi(D, Delta_r):



    # User input ------------------------------------------------------------------
    # radial step size [in microns]
    #Delta_r = input("Radial step size [in microns]: ");
    # diameter of the scan [in microns]
    #D = input("Diameter of the round scan [in microns]: ");
    # center coordinates of the scan
    x_c, y_c = 0., 0.

    # Parameter calculation--------------------------------------------------------

    # number of shells
    N_r = np.ceil(D/(2*Delta_r));
    print 'The scan will have', int(N_r), 'shells,'
    # number of points in first shell, assuming an arc step size of ca. Delta r
    N_theta = np.ceil(2*pi);
    print int(N_theta), 'points in the first shell,'
    # total number of required scan points
    N_tot = N_theta/2*N_r*(N_r+1);
    print 'and a total number of', int(N_tot+1), 'scan points.'
            
    # Coordinate generation--------------------------------------------------------

    # coordinate arrays
    x = sp.zeros(N_tot+1);
    y = sp.zeros(N_tot+1);

    # center point
    x[0] = x_c;
    y[0] = y_c;

    # loop to generate the coordinates
    for k in range(1,int(N_r)+1):
        for m_k in range(1,int(N_theta*k)+1):
            x[N_theta*k/2*(k-1) + m_k] = x_c + Delta_r*k*np.cos((m_k-1)*2*pi/(N_theta*k));
            y[N_theta*k/2*(k-1) + m_k] = y_c + Delta_r*k*np.sin((m_k-1)*2*pi/(N_theta*k));

    return x,y


