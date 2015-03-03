#
# coding: utf-8
# Copyright (C) DATADVANCE, 2010-2014
#
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
import sys


def rrms(x_one, x_two):
    """
    calculate relative root mean square error
    """
    return np.sqrt(np.mean(np.abs(np.power(x_one, 2) - np.power(x_two,2)))) / (np.max(x_two) - np.min(x_two))

def scatter_plot(xs, ys, zs, path_to_save):
  fig = plt.figure()
  ax = Axes3D(fig)
  
  sc = ax.scatter(xs, ys, zs, c='r', marker='o')

  ax.set_xlabel(r'$t_1$', fontsize = 20)
  ax.set_ylabel(r'$t_2$', fontsize = 20)
  ax.set_zlabel('F', fontsize = 20)
  #plt.grid(True)

  plt.suptitle('Compressed points space',fontsize = 25)
  plt.savefig(path_to_save, format='png')
  print ' - Plot is saved to %s' % os.path.join(os.getcwd(), path_to_save)
  if not os.environ.has_key('SUPPRESS_SHOW_PLOTS'):
    print 'Close window to continue script.'
    plt.show()

orig_dim = 5
    
sample_size = 100
X = [[np.random.random()*2-1 for j in range(orig_dim)] for i in range(sample_size)]
def f(x):
    return [np.sum(x) * np.sum(x[0:2]), np.sum(x[0:2])**2]  
F = [f(x) for x in X]

np.savetxt('train_set.txt', np.concatenate((X, F), axis = 1))


'''
6) Plot the compressed space (the 3-dimensional plot of the dependency in reduced-dimensional space)
'''
testX = [[np.random.random()*2-1 for j in range(orig_dim)] for i in range(500)]
testF = [f(x) for x in testX]

np.savetxt('test_set.txt', np.concatenate((testX, testF), axis = 1))
