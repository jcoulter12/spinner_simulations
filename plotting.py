import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import os
import math
from sys import argv
from time import strftime

#script,jobNum = argv
jobNum=10
MSDeta=np.zeros((jobNum,len(np.load('MSDeta0.npy'))))
#MSDtau=np.zeros(jobNum,len(np.open('MSDtau0.npy'))

for i in range(jobNum):
	MSDeta[i,:]=np.load('MSDeta'+str(i)+'.npy')
	#print(MSDeta[i])
	#MSDtau[i,:]=np.load('MSDtau'+str(i)+'.npy')

#Here I will insert the code segments to plot the results of the old code

print("done!")