import numpy as np
from matplotlib import pyplot as plt
#import scipy.io as sio
#import os
#import math
from sys import argv
from time import strftime
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

startTime = datetime.now()

#Define the parameters =================================================
#LATTICE -------------
basis=2
lattice_constant=1
Nposts=50
shift=0.0
tot_posts=Nposts*2*Nposts*2
x_obst1=np.zeros((tot_posts)) 
y_obst1=np.zeros((tot_posts)) 

#TIME ----------------
time_steps=10000
Nspinners=1
MSDSpinners=0
probRes=100.0

#FORCE ---------------
Nres=30
dt=10**-2

script,jobNum, noise, eta, omega = argv
eta=float(eta)
noise=float(noise)
gamma_t=1-eta
omega=float(omega)

#=======================================================================
# Defines the passive particle obstacles for a give array and shift value
#=======================================================================
def lattice_generator():
	xsq1=np.zeros((tot_posts,2)) 
	xsq2=np.zeros((tot_posts,2)) 
	k=0 
	# if shift is 0, it reduces to a typical square lattice
	for i in range(-Nposts,Nposts): 
		for j in range(-Nposts,Nposts): 
			if(i%4==0): #even number column
				if(j%2==0): #even number row
					xsq1[k,:]=(i*a1) + (j*b1)
				if(j%2==1): #odd number row
					xsq1[k,:]=((i+shift)*a1) + ((j+shift+y_shift)*b1)
			if(i%4==1): #odd number column
				if(j%2==0): #even number row
					xsq1[k,:]=((i-shift)*a1) + ((j-shift)*b1)
				if(j%2==1): #odd number row                 
					xsq1[k,:]=(i*a1) + ((j+y_shift)*b1)
			if(i%4==2): #even number column
				if(j%2==0): #even number row
					xsq1[k,:]=((i+shift)*a1) + ((j-shift)*b1)
				if(j%2==1): #odd number row
					xsq1[k,:]=(i*a1) + ((j+y_shift)*b1)
			if(i%4==3): #odd number column
				if(j%2==0): #even number row
					xsq1[k,:]=(i*a1) + (j*b1)
				if(j%2==1): #odd number row
					xsq1[k,:]=((i-shift)*a1) + ((j+shift+y_shift)*b1)
			k+=1   
	np.save('lattice_x.npy', xsq1[:,0]*5)
	np.save('lattice_y.npy', xsq1[:,1]*5)
	return xsq1[:,0]*5, xsq1[:,1]*5

#=======================================================================
# This creates either of the two defined lattice types
#=======================================================================
a1=np.array([1,0])*lattice_constant
b1=np.array([0,1])*lattice_constant
if(basis==0): #Simple cubic primitive vectors
	shift=0
	y_shift=0
if(basis==3): #"Jahn Teller" distorted created by shifting two square lattices
	shift=0.25
	y_shift=0
if(basis==2): 
	shift=0
	y_shift=0.25
x_obst1,y_obst1=lattice_generator()

#=======================================================================
# The solver to run the numerical model 
#=======================================================================
def force_calc(vecx): 
	r_cube1=np.sqrt((vecx[0]-x_obst1)**2+(vecx[1]-y_obst1)**2)
	Fx,Fy=0,0
	xm1=omega*(gamma_t*(vecx[0]-x_obst1)-eta*(vecx[1]-y_obst1)) 
	ym1=omega*(gamma_t*(vecx[1]-y_obst1)+eta*(vecx[0]-x_obst1)) 
	r=np.zeros((len(r_cube1),2))
	#divide the force up by region
	for i in range(len(r_cube1)):
		if(r_cube1[i]<1): #it's close to the post
			if(r_cube1[i]==0):
				r[i,0]=0
				r[i,1]=0
			else:
				r[i,0]=(vecx[0]-x_obst1[i])/r_cube1[i]
				r[i,1]=(vecx[1]-y_obst1[i])/r_cube1[i]
			Fx+=-100*(r_cube1[i]-1)*r[i,0]
			Fy+=-100*(r_cube1[i]-1)*r[i,1]
		if(1<=r_cube1[i]<=35): #it's in the middle
			Fx+=(xm1[i]*(r_cube1[i]**-4))
			Fy+=(ym1[i]*(r_cube1[i]**-4))
		if(r_cube1[i]>35): #it's too far away
			Fx+=0
			Fy+=0
	return Fx,Fy

#=======================================================================
# CALL TO RUN THE NUMERICAL MODEL FOR TRAJECTORY
#=======================================================================
for n in range(Nspinners):
	if(basis==0):
		prob_vals=np.zeros((probRes,probRes))
	if(basis==2):
		prob_vals=np.zeros((2*probRes,probRes))
	x_vec=np.zeros((time_steps,2))
	f_vec=np.zeros((time_steps,2))
	#x_vec[0,0]=np.abs(np.random.randn(1)*3)
	#x_vec[0,1]=np.abs(np.random.randn(1)*3)
	x_vec[0,0]=2
	x_vec[0,1]=2
	for i in range(1,time_steps):
		x_vec[i,:]=x_vec[i-1,:]+f_vec[i-1,:]*(dt)
		x_vec[i,0]+=np.sqrt(dt)*noise*np.random.randn()
		x_vec[i,1]+=np.sqrt(dt)*noise*np.random.randn()
		f_vec[i,0],f_vec[i,1]=force_calc(x_vec[i,:]) 
		#print(x_vec[i,:])
		if(basis==0):
			x=np.floor(np.abs(x_vec[i,0]%5)/(5/probRes))
			y=np.floor(np.abs(x_vec[i,1]%5)/(5/probRes))
		if(basis==2):
			x=np.floor(np.abs(x_vec[i,0]%5)/(5/probRes))
			y=np.floor(np.abs(x_vec[i,1]%10)/(5/(probRes)))
		if(i%1000==0):
			print(i)
		prob_vals[y,x]+=1
		#print("x: " + str(x) + ', ' + str(x_vec[i,1])+ ', y: ' + str(y) + ', ' + str(x_vec[i,1]))
	print(prob_vals)
	plt.figure(figsize=((10,20))) 
	plt.pcolor(prob_vals)
	cm=plt.cm.get_cmap('rainbow')
	plt.savefig("prob_density_"+ str(n) + ".pdf")
	plt.close()
#=======================================================================
# DRAW OUT THE TRAJECTORY IN TIME
#=======================================================================
for n in range(Nspinners):	
	plt.figure(figsize=((10,10))) 
	t=range(time_steps)
	l=plt.scatter(x_obst1,y_obst1,s=30,color="green")
	sc=plt.scatter(x_vec[:,0],x_vec[:,1], 
								c=t, 
								vmin=0, 
								vmax=time_steps, 
								s=30,
								edgecolors='none',
								cmap=cm
								)
plt.xlim(-100,100)
plt.ylim(-100,100)
plt.savefig("traj" + str(n) + ".png")
plt.close()
#=======================================================================
#WRITE OUTPUTS 
#=======================================================================

print("jobNum: " + str(jobNum) + "==========================================================")
print("Run Time : " + str(datetime.now() - startTime))
print("omega: " + str(omega))
print("noise: " + str(noise))
print("Nspinners: " + str(Nspinners))
print("gamma:" + str(gamma_t))
print("timesteps: " + str(time_steps))
print("dt: " + str(dt))
print("Nposts: " + str(Nposts))
print("Nres: " + str(Nres))
print("Lattice_constant: " + str(lattice_constant))
print("OTHER NOTES: " + "gamma=1-eta")
if(basis==3):
	print("             " + "distorted")
else:
	print("             " + "square")
