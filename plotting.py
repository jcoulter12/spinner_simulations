import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import os
import math
from sys import argv
from time import strftime
import matplotlib.cm as cm

jobNum=5
shiftRes=101.0
etaRes=100.0
Nspinners=4
#=======================================================================
# PLOT THE LATTICE 
#=======================================================================
x_obst1=np.load('lattice_x_shift_0.25.npy')
y_obst1=np.load('lattice_y_shift_0.25.npy')
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
	#MSDtau=np.load('MSDshift_0_spinner_0_shift_0.0.npy')
#=======================================================================
# PLOT MSD vs delta tau
#=======================================================================
	# x_vals=np.zeros(len(MSDtau))
	# for u in range(0,len(MSDtau)):
	# 	x_vals[u]=(u+1)*2
	# plt.ylabel('(MSD)')
	# plt.xlabel('(delta tau)')
	# x=np.log10(x_vals[:])
	# x[x_vals[:]==0]=0
	# colors = ['r','g','b']
	# for n in range(0,Nspinners): # number of MSD spinners
	# 	MSDtau=np.load('MSDtau'+str(i)+'spinner_' + str(n) + '.npy')
	# 	y=np.log10(MSDtau[:])
	# 	y[MSDtau[:]==0]=0
	# 	plt.scatter(x,y, color=colors[n])

	# 	z = np.polyfit(x, y, 1)
	# 	p = z[0]*x + z[1] 
	# 	plt.plot(x,p[:],"--",color=colors[n],label=('y=%.6fx+(%.6f)'%(z[0],z[1])))
	# 	ax = plt.gca()
	# 	#ax.text(2,n,('y=%.6fx+(%.6f)'%(z[0],z[1])))
	# 	print("MSDtau Fit: y=%.6fx+(%.6f)"%(z[0],z[1]))

	# 	#do linear fit: log(y) = p(1) * log(x) + p(2)
	# 	p = np.polyfit(x, y, 1)

	# 	#retrieve original parameters
	# 	tau = p[0]
	# 	k = np.exp(p[1])
	# 	#print("MSDtau Exp Fit: y=%.6fx^(%.6f)"%(k,tau))
	# 	#plt.loglog(x_vals, MSDtau[n], '.')
	# 	#plt.loglog(x_vals, k*x_vals**tau, 'r')
	# plt.legend(loc='upper left')
	# plt.savefig("MSDtau_"+ str(i) + ".pdf")
	# plt.close()
#=======================================================================
# PLOT MSD vs delta tau WITH SHIFT
#=======================================================================
	# MSDshift=np.zeros((Nspinners,(shiftRes-1),2))
	# for n in range(0,Nspinners): # number of MSD spinners
	# 	for s in range((int)(shiftRes-1)): #shiftRes range
	# 		shift=s/(shiftRes-1)
	# 		MSDtau=np.load('MSDshift_' + str(i) + '_spinner_' + str(n) + "_shift_" + str(shift) + '.npy')
	# 		y=np.log10(MSDtau[:])
	# 		y[MSDtau[:]==0]=0
	# 		plt.scatter(x,y) 

	# 		#the fit goes here
	# 		z = np.polyfit(x, y, 1)
	# 		p = z[0]*x + z[1] 
	# 		plt.plot(x,p[:],"--",color=colors[n],label=('y=%.6fx+(%.6f)'%(z[0],z[1])))
	# 		print("MSDtau anomaly param, shift of %.1f: m=%.6f"%(shift,z[0]))

	# 		#save the values so that we can plot slope vs shift
	# 		MSDshift[n,s,0]=shift
	# 		MSDshift[n,s,1]=z[0]

	# 	plt.legend(loc='upper left')
	# 	plt.savefig("MSDtau_"+ str(i) + "_spinner#" + str(n)+ ".pdf")
	# 	print("====================================================")
	# 	plt.close()
#=======================================================================
# MSD vs Shift Value
#=======================================================================
	# MSDshift=np.load('MSDshift_0_spinner_0.npy')
	# plt.title('MSD vs Shift for 100 values of shift, 0 -> 1, timesteps=1000, dt = 0.01')
	# plt.scatter((np.arange(0,shiftRes))/100, MSDshift[:])
	# plt.ylabel('MSD')
	# plt.xlabel('shift')
	# plt.savefig("MSDshift_"+ str(i) + ".pdf")
	# plt.close()
#=======================================================================
# MSD Slope vs Shift Value
#=======================================================================
		# plt.scatter(MSDshift[n,:,0],MSDshift[n,:,1],color=colors[n])
		# plt.savefig("MSDshift_"+ str(i) + "_spinner#" + str(n)+ ".pdf")
		# plt.close()
#=======================================================================
# MSD vs Eta
#=======================================================================
	# MSDeta=np.load('MSDeta0spinner_0.npy')
	# plt.scatter((np.arange(etaRes))/(etaRes), MSDeta[:])
	# plt.title('MSD vs Eta for 100 values of eta, 0 -> 1, timesteps=10000, dt = 0.01')
	# plt.ylabel('MSD')
	# plt.xlabel('eta')
	# plt.savefig("MSDeta_"+ str(i) + ".pdf")
	# #print(MSDeta)
	# plt.close()
#=======================================================================
# MSD vs Noise
#=======================================================================
	# MSDnoise=np.load('MSDnoise0spinner_0.npy')
	# plt.scatter(MSDnoise[:,0], MSDnoise[:,1])
	# plt.title('MSD vs Noise for 20 values of noise, 0.0 -> 1, timesteps=10000, dt = 0.001')
	# plt.ylabel('MSD')
	# plt.xlabel('Noise')
	# plt.savefig("MSDnoise_"+ str(i) + ".pdf")
	# plt.close()
#=======================================================================
# DRAW OUT THE TRAJECTORY IN TIME
#=======================================================================
	#time_steps=len(np.load('traj_0_shift_0.npy'))
	time_steps=len(np.load('traj_0_spinner_0.npy'))
	#plt.title("")
	plt.figure(figsize=((10,10))) 
	cm=plt.cm.get_cmap('rainbow')
	t=range(time_steps)
	for n in range(0,Nspinners): # load in all the paths
		path=np.load("traj_" + str(i) + "_spinner_" + str(n) + ".npy")
		#print(path)
		#path=np.load('traj_' + str(i) + '_spinner_' + str(n) + '.npy') 
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
	plt.xlim(-50,50)
	plt.ylim(-50,50)
	plt.savefig("traj" + str(i) + ".png")
	plt.close()
	
print("done!")
