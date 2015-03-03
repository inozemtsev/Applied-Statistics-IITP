#
# coding: utf-8
# Copyright (C) DATADVANCE, 2010-2014
#
# GTOpt Tutorial. Auxilary functions 
def my_plotter(result):
  import matplotlib.pyplot as plt
  import numpy as np
  plt.clf()
  fig = plt.figure(1)
  
  # generated solutions; values() method returns transposed table
  plt.plot(result.converged.f.values()[0], result.converged.f.values()[1], 'r.', label='Pareto frontier (GTOpt)', markersize = 11)
  # true Pareto frontier
  f1_sample = np.linspace(0, 1, 1000)
  plt.plot(f1_sample, 1 - np.sqrt(f1_sample), label='Pareto frontier (analytical)')
  plt.xlabel('f1')
  plt.ylabel('f2')
  plt.legend()


def my_plotter_gso(my_result, my_resultGS, true_function):
  import matplotlib.pyplot as plt
  import numpy as np
  plt.clf()
  fig = plt.figure(1)
  
  # generated solutions; values() method returns transposed table
  plt.plot(my_result.optimal.x[0][0], my_result.optimal.f[0][0], 'g.', label='local optimum', markersize = 11)
  plt.plot(my_resultGS.optimal.x[0][0], my_resultGS.optimal.f[0][0], 'r.', label='global optimum', markersize = 11)
  # true Pareto frontier
  test_sample = np.linspace(0, 1, 1000)
  test_f = [true_function(x) for x in test_sample]
  plt.plot(test_sample, test_f, label='true function')
  plt.xlabel('x')
  plt.ylabel('f(x)')
  plt.legend()
