import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import os
import math
from sys import argv
from time import strftime

jobNum=5
shiftRes=10
#=======================================================================
# PLOT THE LATTICE 
#=======================================================================
x_obst1=np.load('lattice_x.npy')
y_obst1=np.load('lattice_y.npy')
plt.figure(figsize=(10,10))
p1=plt.plot(x_obst1,y_obst1,'o',markersize=15,markeredgewidth=4,color="red")
#if(basis==1):
	#p2=plt.plot(xsq2[:,0]*5,xsq2[:,1]*5,'o',markersize=15,markeredgewidth=4,color="blue")
plt.xlim(-20,20)
plt.ylim(-20,20)
plt.axis('off')
plt.savefig("lattice.pdf")
plt.close()

for i in range (0,jobNum):
	#MSDeta[i,:]=np.load('MSDeta'+str(i)+'.npy')
	MSDtau=np.load('MSDtau0spinner_0.npy')
#=======================================================================
# PLOT MSD vs delta tau
#=======================================================================
	x_vals=np.zeros(len(MSDtau))
	for u in range(0,len(MSDtau)):
		x_vals[u]=(u+1)*2
	plt.ylabel('(MSD)')
	plt.xlabel('(delta tau)')
	x=np.log10(x_vals[:])
	x[x_vals[:]==0]=0
	for n in range(0,3): # number of MSD spinners
		MSDtau=np.load('MSDtau'+str(i)+'spinner_' + str(n) + '.npy')
		y=np.log10(MSDtau[:])
		y[MSDtau[:]==0]=0
		plt.scatter(x,y)

		z = np.polyfit(x, y, 1)
		p = z[0]*x + z[1] 
		plt.plot(x,p[:],"r--")
		print("MSDtau Fit: y=%.6fx+(%.6f)"%(z[0],z[1]))

		#do linear fit: log(y) = p(1) * log(x) + p(2)
		p = np.polyfit(x, y, 1)

		#retrieve original parameters
		tau = p[0]
		k = np.exp(p[1])
		#print("MSDtau Exp Fit: y=%.6fx^(%.6f)"%(k,tau))
		#plt.loglog(x_vals, MSDtau[n], '.')
		#plt.loglog(x_vals, k*x_vals**tau, 'r')
		plt.savefig("MSDtau_"+ str(i) + "_spinner#" + str(n)+ ".pdf")
		plt.close()
#=======================================================================
# PLOT MSD vs delta tau WITH SHIFT
#=======================================================================
#	for p in range(shiftRes): #shiftRes range
#		shift=p/shiftRes
#		MSDtau=np.load('MSDshift' + str(jobNum) + '_spinner_' + str(n) + "_shift_" + str(shift) + '.npy')
#		y=np.log10(MSDtau[n,:])
#		y[MSDtau[n,:]==0]=0
#		plt.scatter(x,y) # add the color in the form of p, a number
#		
#		#put some kind of legend here 
#
#		#the fit goes here
#		z = np.polyfit(x, y, 1)
#		p = z[0]*x + z[1] 
#		plt.plot(x,p[:],"r--")
#		print("MSDtau anomaly param, shift of %.6f: m=%.6fx"%(shift,z[0]))
#	plt.savefig("MSDtau_"+ str(i) + "_spinner#" + str(n)+ ".pdf")
#=======================================================================
# DRAW OUT THE TRAJECTORY IN TIME
#=======================================================================
	Nspinners=5
	time_steps=len(np.load('traj0spinner_0.npy') )
	#plt.title("")
	plt.figure(figsize=((10,10))) 
	cm=plt.cm.get_cmap('rainbow')
	t=range(time_steps)
	for n in range(Nspinners): # load in all the paths
		path=np.load('traj' + str(i) + 'spinner_' + str(n) + '.npy') 
		#plt.quiver(x_vf[:,:,0]/3, x_vf[:,:,1]/3, f_vf[:,:,0], f_vf[:,:,1],      
		#            (np.sqrt(f_vf[:,:,0]**2+f_vf[:,:,1]**2)),                  
		#            cmap=cm,
		#            scale=10000
		#            )
		l=plt.scatter(x_obst1,y_obst1,s=30,color="green")
		sc=plt.scatter(path[:,0],path[:,1], 
									c=t, 
									vmin=0, 
									vmax=time_steps, 
									s=30,
									edgecolors='none',
									cmap=cm
									)
	plt.xlim(-30,30)
	plt.ylim(-30,30)
	plt.savefig("traj" + str(i) + ".png")
	plt.close()
	
print("done!")
