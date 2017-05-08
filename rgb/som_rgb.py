import csv
import random
from som import som
import numpy as numpy
import decimal
import matplotlib.pyplot as plt


input = numpy.array(
         [[1, 0, 0], #red
          [1, 1, 0 ], #yellow
          [0, 1, 0 ], #green
          [0, 1, 1 ], #ciano
          [0, 0, 1 ], #blue
          [1, 0, 1 ]])  #magenta

somCol = 40
somRow = 40

sigmaInitial = 20
radius = 18
maxIterations = 500* (somCol * somRow)


som = som(input,maxIterations ,sigmaInitial,somCol,somRow,radius)
ans = som.trainmodel()


