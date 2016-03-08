######################################################
# plotEllipsoid.
#
# Author: Manuel Diaz Ramos
#
# Plot an ellipsoid given an orthonormal, right handed
# transformation matrix, R and the semi - axis, semi
#
# For the Stat. O.D. project R is made up of the eigenvectors
# of the upper 3x3 portion of the covariance matrix.  semi
# contains sigma_x, sigma_y, sigma_z in a column vector.
#
# Based on 
# http://ccar.colorado.edu/ASEN5070/files/plotEllipsoid.m
# http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
######################################################

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def plotEllipsoid(R, semi):
    
    nn = 256

    phi = np.linspace(0,2*np.pi, nn)    # angle of the projection in the xy-plane
    theta = np.linspace(0, np.pi, nn)   # polar angle
    
    # Creating the mesh
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))    
    
    x = x * semi[0]
    y = y * semi[1]
    z = z * semi[2]
    
    C = np.zeros([nn, nn, 3])
    for i in range(0, nn):
        for j in range(0,nn):
            C[i, j, :] = R.dot(np.array([x[i,j], y[i,j], z[i,j]]))
    
    x = C[:, :, 0]
    y = C[:, :, 1]
    z = C[:, :, 2]
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, cmap=cm.get_cmap())
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    max_radius = np.max([np.max(x), np.max(y), np.max(z)])
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
    
    # Change these angles to rotate the figure
    ax.elev = 25 # [deg]
    ax.azim = 25 # [deg]
    
    return

def plotEllipse(ra, rb, ang, x0, y0, ellipse_label, color, Nb = 256):
    """
    based on matlab code ellipse.m written by D.G. Long,
    Brigham Young University, based on the
    CIRCLES.m original
    written by Peter Blattner, Institute of Microtechnology,
    University of
    Neuchatel, Switzerland, blattner@imt.unine.ch
    Modified by Manuel Diaz Ramos, CU-Boulder, manuel.diazramos@colorado.edu
    :param ra: semi-major axis length.
    :param rb: semi-minor axis length
    :param ang: angle between the semi-major axis and x.
    :param x0: x-position of centre of ellipse
    :param y0: y-position of centre of ellipse
    :param center_point_label:
    :param ellipse_label:
    :param Nb: No. of points that make an ellipse
    :return:
    """
    co = np.cos(ang)
    si = np.sin(ang)
    the = np.linspace(0,2*np.pi,Nb)
    X = ra*np.cos(the)*co - si*rb*np.sin(the) + x0
    Y = ra*np.cos(the)*si + co*rb*np.sin(the) + y0

    plt.plot(X,Y,"b.-",ms=1,label=ellipse_label, color=color)
    #plt.plot(x0, y0, 'or', label=center_point_label, color='g')

    return
