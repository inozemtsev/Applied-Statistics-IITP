#
# coding: utf-8
# Copyright (C) DATADVANCE, 2010-2014
#
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return [np.sum(x) * np.sum(x[0:2]), np.sum(x[0:2])**2]  

def conv57to59(afl):
  afl = np.insert(afl, 30, 0)
  afl = np.insert(afl, 57, afl[0])
  return afl
  
def conv59to57(afl):
  afl = np.delete(afl, 29)
  afl = np.delete(afl, 57)
  return afl

def gen_random_set(minX, maxX, dimension):
  return np.multiply(np.random.rand(dimension), (maxX - minX)*0.8) + minX + 0.1*(maxX-minX)

def rms(x_one, x_two):
    """
    calculate relative root mean square error
    """
    return np.sqrt(np.mean(np.abs(np.power(x_one, 2) - np.power(x_two,2))))

def calculate_errors(values, values_predicted, train_values):
  mean_values = np.tile(np.mean(train_values, axis = 0), (values.shape[0], 1))
  const_error = (np.mean(np.mean((values - mean_values)**2)))**0.5
  
  residuals = np.abs(values - values_predicted)
  RMS = (np.mean(np.mean(residuals**2)))**0.5 
  MAE = np.mean(np.mean(residuals))
  print ' - mean absolute error (MAE) is ' + str(MAE) + ';'
  print ' - root-mean-square error (RMS) is ' + str(RMS) + ';'
  print ' - relative root-mean-square error (RRMS) is ' + str(RMS / const_error) + ';\n'  

def plot_airfoils(ref_points, test_afl_reconstr, compressed_dim, path_to_save):
  plt.plot(ref_points, test_afl_reconstr, label = 'Red. dim = ' + str(compressed_dim))
  plt.legend(loc = 'best')
  plt.title('GTDR Reconstructed Airfoil Example')
  plt.savefig(path_to_save, format='png')
  print ' - Plot is saved to %s' % os.path.join(os.getcwd(), path_to_save)
  if not os.environ.has_key('SUPPRESS_SHOW_PLOTS'):
    print 'Close window to continue script.'
    plt.show()

def comparison_plot(ref_points, new_afl_reconstr, rand_afl, path_to_save):
  plt.figure()  
  plt.plot(ref_points, new_afl_reconstr, label = 'generated airfoil')
  plt.plot(ref_points, rand_afl, label = 'random airfoil')
  plt.legend(loc = 'best')
  plt.title('GTDR Generated Airfoil Example')
  plt.savefig(path_to_save, format='png')
  print ' - Plot is saved to %s' % os.path.join(os.getcwd(), path_to_save)
  if not os.environ.has_key('SUPPRESS_SHOW_PLOTS'):
    print 'Close window to continue script.'
    plt.show()

def rrms(x_one, x_two):
    """
    calculate relative root mean square error
    """
    return np.sqrt(np.mean(np.abs(np.power(x_one, 2) - np.power(x_two,2)))) / (np.max(x_two) - np.min(x_two))

def scatter_plot(xs, ys, zs, path_to_save=None):
  fig = plt.figure()
  ax = Axes3D(fig)
  
  sc = ax.scatter(xs, ys, zs, c='r', marker='o')

  ax.set_xlabel(r'$t_1$', fontsize = 20)
  ax.set_ylabel(r'$t_2$', fontsize = 20)
  ax.set_zlabel('F', fontsize = 20)
  #plt.grid(True)

  plt.suptitle('Compressed points space',fontsize = 25)
  if path_to_save:
    plt.savefig(path_to_save, format='png')
    print ' - Plot is saved to %s' % os.path.join(os.getcwd(), path_to_save)
  if not os.environ.has_key('SUPPRESS_SHOW_PLOTS'):
    print 'Close window to continue script.'
    plt.show()
  return ax

def plot_optimization_results(orig_X, orig_Y, fun, resX1=None, resY1=None, resX2=None, resY2=None):
  ax = scatter_plot(orig_X[:,0],orig_X[:,1],orig_Y)
  ax.plot([-1,1],[1, -1], label='Optimal space')

  X = np.arange(-1, 1, .05)
  Y = np.arange(-1, 1, .05)
  X, Y = np.meshgrid(X, Y)
  Z = map(fun, X, Y)

  ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, label='Surface')
  
  if resX1:
    ax.scatter(resX1[0][0], resX1[0][1], resY1[0][0], c='b', marker='o', label='Original space solution', s=50)
    ax.plot([],[],' ', c='b', marker='o', label='Original space solution', markersize=8)

  if resX2:
    ax.scatter(resX2[0][0], resX2[0][1], resY2[0][0], c='r', marker='o', label='Reduced space solution', s=50)
    ax.plot([],[],' ', c='r', marker='o', label='Reduced space solution', markersize=8)

  plt.legend(loc='best')



def scatter_plot_wine(xs, ys, zs, color=None):
  fig = plt.figure()
  ax = Axes3D(fig)
  
  sc = ax.scatter(xs, ys, zs, c=color, marker='o')

  ax.set_xlabel(r'$t_1$', fontsize = 20)
  ax.set_ylabel(r'$t_2$', fontsize = 20)
  ax.set_zlabel(r'$t_3$', fontsize = 20)
  #plt.grid(True)

  plt.suptitle('Compressed points space',fontsize = 25)
  if not os.environ.has_key('SUPPRESS_SHOW_PLOTS'):
    print 'Close window to continue script.'
    plt.show()
