#
# coding: utf-8
# Copyright (C) DATADVANCE, 2010-2014
#
from pylab import *
import os
from mpl_toolkits.mplot3d import Axes3D

def trainGTApproxModel(highFidelityTrainPoints, highFidelityTrainValues, options=dict()):
  '''
  Build GTApprox model 
  '''
  # create gtapprox builder
  builder = gtapprox.Builder()
  # set logger
  logger = loggers.StreamLogger()
  builder.set_logger(logger)
  # set options if available
  builder.options.set(options)
  
  return builder.build(highFidelityTrainPoints, highFidelityTrainValues)

def trainGTDFModel(highFidelityTrainPoints, highFidelityTrainValues, lowFidelityTrainPoints, lowFidelityTrainValues, options=dict()):
  '''
  Build GTDF model 
  '''
  # create gtdf builder
  builder = gtdf.Builder()
  # set logger
  logger = loggers.StreamLogger()
  builder.set_logger(logger)
  # set options if available
  builder.options.set(options)
  
  return builder.build(highFidelityTrainPoints, highFidelityTrainValues, 
                       lowFidelityTrainPoints, lowFidelityTrainValues)

def buildModels(lowFidelityTrainPoints, lowFidelityTrainValues, 
                highFidelityTrainPoints, highFidelityTrainValues):
  '''
  Build surrogate models.
  GT Approx Gp and GT DF VFGP techniques are used for surrogate model construction
  '''
  
  # set options
  gtaOptions = {
  'GTApprox/Technique': 'GP',
  'GTApprox/LogLevel': 'Info',
  }
  # get gt approx model
  gtaModel = trainGTApproxModel(highFidelityTrainPoints, highFidelityTrainValues, 
                                gtaOptions)
  
  # set options
  gtdfOptions = {
  'GTDF/Technique': 'VFGP',
  'GTDF/LogLevel': 'Info',
  }
  # get gt df model 
  gtdfModel = trainGTDFModel(highFidelityTrainPoints, highFidelityTrainValues, 
                             lowFidelityTrainPoints, lowFidelityTrainValues, 
                             gtdfOptions)  
 
  return gtaModel, gtdfModel

def getTestData(sampleSize):
  '''
  Generate test data.
  '''
  points = np.reshape(np.linspace(0., 1., sampleSize), (sampleSize, -1))
  
  lowFidelityValues = lowFidelityFunction(points)
  highFidelityValues = highFidelityFunction(points)
  
  return points, lowFidelityValues, highFidelityValues

def calculateValues(testPoints, gtaModel, gtdfModel):
  '''
  Calculate models on given sample.
  '''
  gtaValues = gtaModel.calc(testPoints)
  gtdfValues = gtdfModel.calc(testPoints)

  return gtaValues, gtdfValues

def plotTrain(lowFidelityTrainPoints, lowFidelityTrainValues, 
               highFidelityTrainPoints, highFidelityTrainValues):
  '''
  Visualize training sample.
  '''
  plt.plot(lowFidelityTrainPoints, lowFidelityTrainValues, 's', 
           markersize = 6.5, markeredgewidth = 2.0,
           markerfacecolor = 'none',  markeredgecolor = 'm')
  plt.plot(highFidelityTrainPoints, highFidelityTrainValues, 'o', 
           markersize = 6.5, markeredgewidth = 2.0,
           markerfacecolor = 'none',  markeredgecolor = 'b')
  
def plotTest(testPoints, lowFidelityTestValues, highFidelityTestValues):
  '''
  Visualize test sample.
  '''
  plt.plot(testPoints, lowFidelityTestValues, 'm', linestyle = '--', linewidth = 2.0, label = 'Low fidelity function')  
  plt.plot(testPoints, highFidelityTestValues, 'b', linestyle = '--', linewidth = 2.0, label = 'High fidelity function')  
  
def plotApproximations(testPoints, gtaValues, gtdfValues):
  '''
  Visualize approximations.
  '''
  plt.plot(testPoints, gtaValues, '-.g', linewidth = 2.0, label = 'GTApprox GP')
  plt.plot(testPoints, gtdfValues, 'r', linewidth = 2.0, label = 'GTDF VFGP')

def showPlots():
  '''
  Configure, show and save plots.
  '''
  plt.xlabel(r'Points', fontsize = 20)
  plt.ylabel(r'Values', fontsize = 20)
  plt.grid(True)
  plt.title('GTDF example')
  plt.legend(loc = 'best')
  name = 'gtdf_simple_example'
  plt.savefig(name)
  print 'Plot is saved to %s.png' % os.path.join(os.getcwd(), name)
  if not os.environ.has_key('SUPPRESS_SHOW_PLOTS'):
    plt.show()

def getTrainDataExactFit(lowFidelityFunction, highFidelityFunction, lowFidelitySampleSize, highFidelitySampleSize):    
  lowFidelityPoints = np.random.rand(lowFidelitySampleSize, 2)
  highFidelityPoints = np.random.rand(highFidelitySampleSize, 2)  
      
  lowFidelityValues = lowFidelityFunction(lowFidelityPoints) + 0.9 * np.random.randn(lowFidelitySampleSize, 1)
  highFidelityValues = highFidelityFunction(highFidelityPoints) + 0.4 * np.random.randn(highFidelitySampleSize, 1)

  return lowFidelityPoints, lowFidelityValues, highFidelityPoints, highFidelityValues



# get test data with 2D-grid
def getTestDataExactFit(lowFidelityFunction, highFidelityFunction, meshGrid):
  [x, y] = np.meshgrid(meshGrid, meshGrid)
  points = np.ones((np.prod(y.shape), 2))
  points[:, 0] = reshape(x, np.prod(x.shape), 1)
  points[:, 1] = reshape(y, np.prod(x.shape), 1)
  
  lowFidelityValues = lowFidelityFunction(points)
  highFidelityValues = highFidelityFunction(points)
  
  return points, lowFidelityValues, highFidelityValues

def getResults(interpolationModel, approximationModel, highFidelityTrainPoints, highFidelityTrainValues, lowFidelityFunction, highFidelityFunction):
  '''
  Generate test set to compare performance:
  '''
  # Get test set
  meshSize = 40
  meshGrid = np.linspace(0, 1, meshSize)
  testPoints, lowFidelityTestValues, highFidelityTestValues = getTestDataExactFit(lowFidelityFunction, highFidelityFunction, meshGrid)
  
  # Get values for test set  
  interpolationValues = interpolationModel.calc(testPoints)
  approximationValues = approximationModel.calc(testPoints)
  
  # Plot results using three subplots
  figureHandle = plt.figure(figsize=(20,15))
  [x, y] = np.meshgrid(meshGrid, meshGrid)

  print ('=' * 60), '\n'
  print 'The first 3D plot corresponds to true function, the second one is obtained using exact fit model and the last one is obtained using non-exact fit model...'
  # True function values
  rect = figureHandle.add_subplot(2, 2, 1).get_position()    
  ax = Axes3D(figureHandle, rect)
  z = (np.array(highFidelityTestValues).reshape(meshSize, meshSize)).transpose()
  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)
  ax.plot3D(highFidelityTrainPoints[:, 0], highFidelityTrainPoints[:, 1], highFidelityTrainValues[:, 0], 
            'o', markersize = 6.0, label='Training set with noise')
  title('True function') 
  ax.legend(loc='lower left')

  # Approximation function values
  rect = figureHandle.add_subplot(2, 2, 2).get_position()    
  ax = Axes3D(figureHandle, rect)
  z = (np.array(approximationValues).reshape(meshSize, meshSize)).transpose()
  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet) 
  ax.plot3D(highFidelityTrainPoints[:, 0], highFidelityTrainPoints[:, 1], highFidelityTrainValues[:, 0], 
            'o', markersize = 6.0)
  title('Approximation') 
  
  # Exact fit function values
  rect = figureHandle.add_subplot(2, 2, 3).get_position()
  ax = Axes3D(figureHandle, rect)
  z = (np.array(interpolationValues).reshape(meshSize, meshSize)).transpose()
  ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet) 
  ax.plot3D(highFidelityTrainPoints[:, 0], highFidelityTrainPoints[:, 1], highFidelityTrainValues[:, 0], 
            'o', markersize = 6.0)
  title('Exact fit')

  # save figure and show it
  name = 'example_gtdf_exact_fit.png'
  plt.savefig(name)
  print 'Plot is saved to %s' % os.path.join(os.getcwd(), name)
  if not os.environ.has_key('SUPPRESS_SHOW_PLOTS'):
    print 'Close window to finish script.'
    plt.show()
