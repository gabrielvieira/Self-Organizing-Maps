import csv
import random
from som import som
import numpy as numpy
import matplotlib.pyplot as plt

#open dataset
csvfile = open('../iris/iris_shuffled.csv', 'rt')
lines = csv.reader(csvfile)
dataset = list(lines)

#convert to numpy and remove class collun
data = numpy.array(dataset)
data = numpy.delete(data, numpy.s_[-1], axis=1)  
data = data.astype(numpy.float) 

#min neurons 3x10 = 30
somCol = 6
somRow = 6

sigmaInitial = 3
radius = 3

# maxIterations = 500
maxIterations = 500*(somRow*somCol)

som = som(data,maxIterations ,sigmaInitial,somCol,somRow,radius)
ans = som.trainmodel()


# get trained newtork and plot image

# input = numpy.array(
# 		[[1, 0, 0], #red
# 		[1, 1, 0 ], #yellow
# 		[0, 1, 0 ], #green
# 		[0, 1, 1 ], #ciano
# 		[0, 0, 1 ], #blue
# 		[1, 0, 1 ]])  #magenta



class_mapping = {

    0 : [1,0,0], #red
    1 : [0,1,0], #green
    2 : [0,0,1]  #blue
}


csvfile = open('../iris/iris_shuffled.csv', 'rt')
lines = csv.reader(csvfile)
dataset = list(lines)
original_data = numpy.array(dataset)

outputImage = numpy.zeros(shape=(somCol,somRow,3))

count = 1;
inputsSize = len(data[:,0])
while(count < inputsSize):

	selectedWeightVector = data[count,:]

	mineuclideanD=numpy.linalg.norm(selectedWeightVector-ans[0,0,:])
	minr=0
	minc=0
	for num in range (0,somRow):
	    for iter in range (0,somCol):

	        temp  = numpy.linalg.norm(selectedWeightVector-ans[num,iter,:])
	        
	        if(temp <=mineuclideanD):
	            minr=num
	            minc=iter
	            mineuclideanD=temp


	outputImage[minr , minc, :] = class_mapping[ int(original_data[count, -1]) ]	            

	count += 1


plt.ion()
plt.imshow(outputImage, interpolation="nearest") 
while True:
	plt.pause(0.01)

# print 'trained model is',ans

# print data


