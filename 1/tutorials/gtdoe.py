#
# coding: utf-8
# Copyright (C) DATADVANCE, 2010-2014
#
from da.macros import gtdoe, gtapprox
from da.macros.loggers import StreamLogger

import matplotlib.pyplot as plt
import numpy as np

import os
import random as rnd

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
'''
Define the dependency::
'''
#[2]
def quadraticFunction(points):
  values = []
  for point in points:
    values.append((point[0] - point[1]) ** 2. + 0.3 * (2 * rnd.random() - 1))
  return values
#[2]


'''
Define generic function for optimal DoE generation. Note that Model type set to 'quadratic' and the log level set to 'Info'::
'''
#[3]
def get_doe(number_points, lb, ub, doeType):  
  # create generator
  generator = gtdoe.Generator()
  # set logger
  generator.set_logger(StreamLogger())

  # set options
  options = {
    'GTDoE/Technique': 'OptimalDesign',
    'GTDoE/OptimalDesign/Type': doeType,
    'GTDoE/OptimalDesign/Model': 'quadratic',
    'GTDoE/Deterministic': 'yes',
    'GTDoE/LogLevel': 'Info'
  }
  generator.options.set(options)
  
  result = generator.generate(bounds=(lb, ub), count=number_points)
  points = result.points
  return points 
#[3]

'''
Define auxiliary generic function for DoE plotting::
'''

#[4]
def plot_doe(points, technique):
  points = np.array(points)

  params = {'legend.fontsize': 26,
            'legend.linewidth': 2}
  plt.rcParams.update(params)

  print "Plotting figures..."
  dim = 2
  doeFigure = plt.figure(figsize = (16, 10))
  plt.plot(points[:, 0], points[:, 1], 'o', markersize = 9, label = technique)
  plt.legend(loc = 'best')
  plt.title('Technique: %s' % technique, fontsize = 28)
  plt.xlim(-1.1, 1.1)
  plt.ylim(-1.1, 1.1)
  plt.ylabel('y', fontsize = 26)
  plt.xlabel('x', fontsize = 26)
  name = 'gtdoe_example_' + technique
  plt.savefig(name)
  print 'Plots are saved to %s.png' % os.path.join(os.getcwd(), name)
#[4]

'''
Define function for full grid generation (for plotting purposes)::
'''

#[5]
# get test data with 2D-grid
def getTestData(targetFunction, meshGrid):
  [x, y] = np.meshgrid(meshGrid, meshGrid)
  points = np.ones((np.prod(y.shape), 2))
  points[:, 0] = np.reshape(x, np.prod(x.shape), 1)
  points[:, 1] = np.reshape(y, np.prod(x.shape), 1)
  
  values = targetFunction(points)
  
  return points, values
#[5]

'''
Define random DoE generation function::
'''

#[6]
def executeRandomDoE(numberPoints, dimension):
  doeGenerator = gtdoe.Generator()
  doeGenerator.options.set('GTDoE/Seed', '101')
  doeGenerator.options.set('GTDoE/Deterministic', True)
  doeGenerator.options.set('GTDoE/Technique', 'RandomSeq')

  bounds = ([-1. for _ in xrange(dimension)], [1. for _ in xrange(dimension)])  

  points = doeGenerator.generate(bounds=bounds, count=numberPoints).points

  return np.array(points)
#[6]


'''
Define function for plotting and saving the resulting approximations. 
The first 3D plot corresponds to true function, the second one is approximation abtained using random design, the third -- obtained using I-optimal design, the last -- using D-optimal design.
'''
#[7]
def plotResults(iOptimalModel, dOptimalModel, randomModel, 
  trainPoints1, trainValues1, trainPoints2, trainValues2,randomPoints, randomValues):
  '''
  Generate test set to compare performance:
  '''
  targetFunction = quadraticFunction
  zeroValues = [0] * len(trainPoints1)
  trainPoints1 = np.array(trainPoints1)
  trainPoints2 = np.array(trainPoints2)
  # Get test set
  meshSize = 30
  meshGrid = np.linspace(-1, 1, meshSize)
  testPoints, testValues = getTestData(targetFunction, meshGrid)

  randomTestValues = randomModel.calc(testPoints)

  # Get values for test set  
  iOptimalValues = iOptimalModel.calc(testPoints)
  dOptimalValues = dOptimalModel.calc(testPoints)
  
  # Plot results using three subplots
  figureHandle = plt.figure()
  [x, y] = np.meshgrid(meshGrid, meshGrid)
  
  
  # True function
  rect = figureHandle.add_subplot(2, 2, 1).get_position()    
  ax = Axes3D(figureHandle, rect)
  z = (np.array(testValues).reshape(meshSize, meshSize)).transpose()
  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)
  plt.title('True function') 

  # Random design and approximation
  rect = figureHandle.add_subplot(2, 2, 2).get_position()    
  ax = Axes3D(figureHandle, rect)
  z = (np.array(randomTestValues).reshape(meshSize, meshSize)).transpose()
  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet) 
  ax.plot3D(randomPoints[:, 0], randomPoints[:, 1], zeroValues, 'o', markersize = 6.0)
  plt.title('Random design and approximation') 

  # D-optimal design and approximation
  rect = figureHandle.add_subplot(2, 2, 3).get_position()    
  ax = Axes3D(figureHandle, rect)
  z = (np.array(dOptimalValues).reshape(meshSize, meshSize)).transpose()
  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet) 
  ax.plot3D(trainPoints1[:, 0], trainPoints1[:, 1], zeroValues, 'o', markersize = 6.0)
  plt.title('D-optimal design and approximation') 
  
  # IV-optimal design and approximation
  rect = figureHandle.add_subplot(2, 2, 4).get_position()    
  ax = Axes3D(figureHandle, rect)
  z = (np.array(iOptimalValues).reshape(meshSize, meshSize)).transpose()
  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet) 
  ax.plot3D(trainPoints2[:, 0], trainPoints2[:, 1], zeroValues, 'o', markersize = 6.0)
  plt.title('IV-optimal design and approximation') 
#[7]


# generate initial train data
def getInitialData(targetFunction):    
  #initialPoints = np.array([0.0111, 0.2718, 0.4383, 0.7511, 0.9652])
  initialPoints = [[0.0111], [0.2718], [0.4383], [0.7511], [0.9652]]
  #initialValues = targetFunction(initialPoints)
  initialValues = targetFunction(initialPoints)
  return initialPoints, initialValues

def trainGpModel(trainPoints, trainValues):
  # create builder
  builder = gtapprox.Builder()
  # setup options
  options = {
  'GTApprox/Technique': 'GP',
  'GTApprox/LogLevel': 'Fatal',
  'GTApprox/AccuracyEvaluation': 'On'
  }
  builder.options.set(options)
  # train GT Approx GP model
  return builder.build(trainPoints, trainValues)

# build approximation models  
def executeAdaptiveDoE(initialPoints, initialValues, numberNewPoints, targetFunction):
  '''
  To perform adaptive DoE technique we need to create blackbox for target function
  '''

  print 'Create blackbox for target function'
  # set blackbox
  targetFunctionBlackBox = TargetBlackBox(targetFunction)


  print 'Perform adaptive DoE process using Maximum Variance criterion'
  # perform adaptive DoE process 
  doeGenerator = gtdoe.Generator()

  trainPoints = doeGenerator.generate(budget=numberNewPoints, bounds=([0.0], [1.0]), 
                                      init_x = initialPoints, init_y = initialValues, 
                                      blackbox = targetFunctionBlackBox, 
                                      options={'GTDoE/Deterministic' : 'on',
                                               'GTDoE/Adaptive/Criterion': 'MaximumVariance'}).points #IntegratedMseGainMaxVar

  trainValues = targetFunction(trainPoints)

  return trainPoints, trainValues


# generate test data    
def getTestDataAdaptive(numberTestPoints, targetFunction):
  points = np.reshape(np.linspace(0., 1., numberTestPoints), (numberTestPoints, 1))
  
  listPoints = []
  for point in points:
    listPoints.append(list(point))

  points = listPoints
  values = targetFunction(points)
 
  return points, values
  
# compare techniques performance
def getResults(trainPoints, trainValues, targetFunction, numberTestPoints):
                   
  # get test set
  testPoints, testValues = getTestDataAdaptive(numberTestPoints, targetFunction)
  
  print 'Construct approximations using points, obtained by adaptive DoE'
  numberInitialPoints = 5;
  numberPoints = len(trainPoints)
  numberNewPoints = numberPoints - numberInitialPoints

  # calculate approximation for test points for each iteration of DoE
  print ('=' * 60), '\n'
  print 'Compare approximation quality using 1D approximation plot'
  print 'Plot approximations for every iteration of adaptive DoE algorithm...'
  for currentPointIndex in range(numberNewPoints):
    print currentPointIndex
    currentTrainPoints = trainPoints[0:(numberInitialPoints + currentPointIndex)]
    currentTrainValues = trainValues[0:(numberInitialPoints + currentPointIndex)]
    
    currentGpModel = trainGpModel(currentTrainPoints, currentTrainValues)
    
    currentPredictedValues = currentGpModel.calc(testPoints)

    currentAe = currentGpModel.calc_ae(testPoints)
  

    # plot results
    params = {'legend.fontsize': 20,
              'legend.linewidth': 2}
    plt.rcParams.update(params)

    figureHandle = figure()
    trainPointsLine, = plt.plot(currentTrainPoints, currentTrainValues, 'ob', markersize = 7.0, linewidth = 2.0)
    trueFunctionLine, = plt.plot(testPoints, testValues, 'b', linewidth = 2.0)
    predictedLine, = plt.plot(testPoints, currentPredictedValues, 'r', linewidth = 2.0)    
    aeLine, = plt.plot(testPoints, currentAe - 10, 'g', linewidth = 2.0) 
    newPointLine, = plt.plot(trainPoints[numberInitialPoints + currentPointIndex], 
                        trainValues[numberInitialPoints + currentPointIndex], 
                        'or', markersize = 7.0, linewidth = 2.0)
    xlabel(r'$x$', fontsize = 30)
    ylabel(r'$y(x)$', fontsize = 30)
    grid(True)
    plt.legend((trainPointsLine, newPointLine, trueFunctionLine, predictedLine, aeLine), 
           ('TrainPoints', 'NewPoint', 'TrueFunction', 'Approximation', 'AccuracyEvaluation'), 'upper left')
         
    plt.title("$f(x) = (6 x - 2)^2 \sin(12 x - 4)$, " + str(currentPointIndex + 1) + " $iteration$", fontsize = 26)

def plot(models, samples, target_function):
  """
  Show and save example plots.

  Args:
    models:  GTApprox models trained using the adaptive and space-filling DOE
             samples (tuple).
    samples: Training samples (tuple: adaptive, space-filling).

  Creates two figures: 3D surface plots of the example function and both models
  and two contour plots showing DoE points distribution in the adaptive and
  space-filling sampling.

  Figures are saved to the script working directory.
  """
  model_a, model_sf = models
  (x_train_a, f_train_a), (x_train_sf, f_train_sf) = samples

  print "Generating plot data. Please wait..."
  # Generate surface plots data.
  x1 = x2 = np.arange(0, 1, 0.02)
  x1, x2 = np.meshgrid(x1, x2)
  f = target_function([x1, x2])
  pts = zip(x1.flat, x2.flat)
  f_a = model_a.calc(pts).reshape(x1.shape)
  f_sf = model_sf.calc(pts).reshape(x1.shape)
  # Prepare scatter plots data.
  xs_a, ys_a = zip(*x_train_a)
  zs_a = zip(*f_train_a)[0]
  xs_sf, ys_sf = zip(*x_train_sf)
  zs_sf = zip(*f_train_sf)[0]

  # Add figure: 3D plots of the original function and both models.
  fig = plt.figure(figsize=(8, 24))
  fig.suptitle("Approximation models: adaptive and space-filling sampling", fontsize=18)
  # Original function, 3D plot.
  ax = fig.add_subplot(311, projection="3d")
  ax.set_title("Original function")
  ax.set_xlabel("$x_1$", fontsize ="16")
  ax.set_ylabel("$x_2$", fontsize ="16")
  ax.set_zlabel("$f(\\overline{x})$", fontsize ="16")
  ax.view_init(35, -65)
  ax.plot_surface(x1, x2, f, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.0, alpha=0.5, antialiased=False)
  # Adaptive sampling model, 3D plot.
  ax = fig.add_subplot(312, projection="3d")
  ax.set_title("Adaptive sampling")
  ax.set_xlabel("$x_1$", fontsize ="16")
  ax.set_ylabel("$x_2$", fontsize ="16")
  ax.set_zlabel("$\\widehat{f}_a(\\overline{x})$", fontsize ="16")
  ax.view_init(35, -65)
  ax.plot_surface(x1, x2, f_a, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.0, alpha=0.5, antialiased=False)
  # Add training sample scatter.
  ax.plot(xs_a, ys_a, zs=zs_a, c="r", marker="o", ms=5, ls="")
  # Spacefilling sampling model, 3D plot.
  ax = fig.add_subplot(313, projection="3d")
  ax.set_title("Spacefilling sampling")
  ax.set_xlabel("$x_1$", fontsize ="16")
  ax.set_ylabel("$x_2$", fontsize ="16")
  ax.set_zlabel("$\\widehat{f}_{sf}(\\overline{x})$", fontsize ="16")
  ax.view_init(35, -65)
  ax.plot_surface(x1, x2, f_sf, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.0, alpha=0.5, antialiased=False)
  # Add training sample scatter and legend.
  scatter = ax.plot(xs_sf, ys_sf, zs=zs_sf, c="r", marker="o", ms=5, ls="")
  plt.figlegend(scatter, ["training sample points"], "lower right")
  # Save model plots.
  filename = "doe_adaptive_models3D"
  fig.savefig(filename)
  print "Model plots saved to %s.png" % os.path.join(os.getcwd(), filename)

  # Add figure: contour plots with DoE points distribution.
  fig = plt.figure(figsize=(16, 8))
  fig.suptitle("Design of experiments", fontsize=18)
  # Adaptive DoE.
  ax = fig.add_subplot(121)
  ax.set_title("Adaptive DoE")
  ax.set_xlabel("$x_1$", fontsize ="16")
  ax.set_ylabel("$x_2$", fontsize ="16")
  contour = ax.contour(x1, x2, f)
  ax.clabel(contour, inline=1, fontsize=10)
  ax.scatter(xs_a, ys_a, c="r")
  # Spacefilling DoE.
  ax = fig.add_subplot(122)
  ax.set_title("Space-filling DoE (LHS)")
  ax.set_xlabel("$x_1$", fontsize ="16")
  ax.set_ylabel("$x_2$", fontsize ="16")
  contour = ax.contour(x1, x2, f)
  ax.clabel(contour, inline=1, fontsize=10)
  ax.scatter(xs_sf, ys_sf, c="r")
  # Save DoE plots.
  filename = "doe_adaptive_scatter"
  fig.savefig(filename)
  print "DoE plots saved to %s.png" % os.path.join(os.getcwd(), filename)

  # Show plots if we may.
  if not os.environ.has_key("SUPPRESS_SHOW_PLOTS"):
    print "Close plot windows to finish."
    plt.show()
