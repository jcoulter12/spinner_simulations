import numpy as np
#from matplotlib import pyplot as plt
#import scipy.io as sio
import os
import math
from sys import argv
from time import strftime
from datetime import datetime
startTime = datetime.now()

#Define the parameters =================================================
#LATTICE -------------
basis=0
lattice_constant=1
Nposts=50

#TIME ----------------
time_steps=10000
Nspinners=5

#FORCE ---------------
alpha=1 #weight parameter for ym and xm later
eta=0.1
#omega=100.0
Nres=30
dt=10**-3
noise=0
etaRes=0.01

script,jobNum, noise, eta, omega = argv
eta=float(eta)
noise=float(noise)
gamma_t=1-eta
omega=float(omega)
#to be sure the path is visible afterwards
if(omega==1000):
    time_steps=time_steps/5
#=======================================================================
# This creates the lattice
#=======================================================================
#Simple cubic primitive vectors
if(basis==0):
    a1=np.array([1,0])*lattice_constant
    b1=np.array([0,1])*lattice_constant
#+++++++++++++++++++++++++++++++++++++++++++++++
#Honeycomb
#created by phase shifting two hexagonal lattices
if(basis==1):
    a1=np.array([0.5,np.sqrt(3)/2])*lattice_constant
    b1=np.array([0.5,-np.sqrt(3)/2])*lattice_constant
    a2=np.array([0.5,np.sqrt(3)/2])*lattice_constant
    b2=np.array([0.5,-np.sqrt(3)/2])*lattice_constant
    x_shift=0.5
    y_shift=1
#+++++++++++++++++++++++++++++++++++++++++++++++
#Hexagonal primitive vectors
if(basis==2):
    a1=np.array([1,0])*lattice_constant
    b1=np.array([0.5,np.sqrt(3)/2])*lattice_constant
#+++++++++++++++++++++++++++++++++++++++++++++++
#Jahn Teller distorted -- type 1
#created by phase shifting two rectangular lattices
if(basis==3):
    a1=np.array([1,0])*lattice_constant
    b1=np.array([0,1])*lattice_constant
    a2=np.array([1,0])*lattice_constant
    b2=np.array([0,1])*lattice_constant
    x_shift=0.25
    y_shift=1
#+++++++++++++++++++++++++++++++++++++++++++++++
#Cubic/rectangular, but using different intensities to create a non-bravais lattice?
#+++++++++++++++++++++++++++++++++++++++++++++++
tot_posts=Nposts*2*Nposts*2
xsq1=np.zeros((tot_posts,2)) 
xsq2=np.zeros((tot_posts,2)) 
k=0 
if(basis==3): #for Jahn Teller
    for i in range(-Nposts,Nposts): 
        for j in range(-Nposts,Nposts): 
            if(i%4==0): #even number column
                if(j%2==0): #even number row
                    #xsq1[k,:]=((i-0.25)*a1) + ((j+0.25)*b1)
                    xsq1[k,:]=(i*a1) + (j*b1)
                if(j%2==1): #odd number row
                    xsq1[k,:]=((i+x_shift)*a1) + ((j+x_shift)*b1)
            if(i%4==1): #odd number column
                if(j%2==0): #even number row
                    xsq1[k,:]=((i-x_shift)*a1) + ((j-x_shift)*b1)
                if(j%2==1): #odd number row                 
                    xsq1[k,:]=(i*a1) + (j*b1)
                    #xsq1[k,:]=((i+0.25)*a1) + ((j-0.25)*b1)
            if(i%4==2): #even number column
                if(j%2==0): #even number row
                    xsq1[k,:]=((i+x_shift)*a1) + ((j-x_shift)*b1)
                if(j%2==1): #odd number row
                    xsq1[k,:]=(i*a1) + (j*b1)
                    #xsq1[k,:]=((i-0.25)*a1) + ((j-0.25)*b1)
            if(i%4==3): #odd number column
                if(j%2==0): #even number row
                    #xsq1[k,:]=((i+0.25)*a1) + ((j+0.25)*b1)
                    xsq1[k,:]=(i*a1) + (j*b1)
                if(j%2==1): #odd number row
                    xsq1[k,:]=((i-x_shift)*a1) + ((j+x_shift)*b1)         
            k+=1 
#for all other lattice types
else:
    for i in range(-Nposts,Nposts): #counts out columns
        for j in range(-Nposts,Nposts): #counts out rows
            xsq1[k,:]=i*a1+j*b1 
            if(basis==1): #for honeycomb
                xsq2[k,:]=((i*a2+x_shift)+(j*b2+y_shift))
            k+=1
x_obst1=xsq1[:,0]*5
y_obst1=xsq1[:,1]*5
x_obst2=np.zeros((len(x_obst1),2))
y_obst2=np.zeros((len(y_obst1),2))
'''
#=======================================================================
# visualizing the lattice
#=======================================================================
plt.figure(figsize=(10,10))
x_obst1=xsq1[:,0]*5
y_obst1=xsq1[:,1]*5
x_obst2=np.zeros((len(x_obst1),2)) 
y_obst2=np.zeros((len(y_obst1),2)) 
p1=plt.plot(xsq1[:,0]*5,xsq1[:,1]*5,'o',markersize=15,markeredgewidth=4,color="red")
if(basis==1):
    x_obst2=xsq2[:,0]*5
    y_obst2=xsq2[:,1]*5
    #p2=plt.plot(xsq2[:,0]*5,xsq2[:,1]*5,'o',markersize=15,markeredgewidth=4,color="blue")     
plt.xlim(-20,20)
plt.ylim(-20,20)
plt.axis('off')
plt.savefig("lattice.pdf")
plt.close()
'''
#=======================================================================
# The solver to run the numerical model 
#=======================================================================

def force_calc(vecx): 
    r_cube1=np.sqrt((vecx[0]-x_obst1)**2+(vecx[1]-y_obst1)**2)
    r_cube2=np.sqrt((vecx[0]-x_obst2)**2+(vecx[1]-y_obst2)**2)
    Fx,Fy=0,0
  
    xm1=omega*(gamma_t*(vecx[0]-x_obst1)-eta*alpha*(vecx[1]-y_obst1)) 
    ym1=omega*(gamma_t*(vecx[1]-y_obst1)+eta*alpha*(vecx[0]-x_obst1)) 
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
    if(basis==2): 
        xm2=omega*(gamma_t*(vecx[0]-x_obst2)-eta*alpha*(vecx[1]-y_obst2)) 
        ym2=omega*(gamma_t*(vecx[1]-y_obst2)+eta*alpha*(vecx[0]-x_obst2)) 
        Fx= Fx + np.sum(xm2*r_cube2) 
        Fy= Fy + np.sum(ym2*r_cube2)
    return Fx,Fy

#=======================================================================
# MEAN SQUARE DISPLACEMENT
#=======================================================================
def force_calc_stub():
    x_ens=np.zeros((Nspinners,time_steps,2))
    x_vec=np.zeros((time_steps,2))
    f_vec=np.zeros((time_steps,2))
    x_vec[0,:]=np.random.randn(2)*10 
    for i in range(1,time_steps):
        #x_vec[i,:]=x_vec[i-1,:]+f_vec[i-1,:]*np.sqrt(dt)
        x_vec[i,:]=x_vec[i-1,:]+f_vec[i-1,:]*(dt)
        x_vec[i,0]+=np.sqrt(dt)*noise*np.random.randn()
        x_vec[i,1]+=np.sqrt(dt)*noise*np.random.randn()
        f_vec[i,0],f_vec[i,1]=force_calc(x_vec[i,:])
        x_ens[0,:,:]=x_vec
    return x_ens
#========================================================================
Nspinners=3
tauRes=100
MSDtau=np.zeros((Nspinners,tauRes))
MSDeta=np.zeros((Nspinners,(int)(1/etaRes)))

for i in range(0,Nspinners):
    '''
    #Eta values -------------------------------- 
    for j in range(0,(int)(1/etaRes)): 
        tau=1
        eta=j*etaRes
        gamma=1-eta
        x_path=force_calc_stub()
        for N in range(10,time-tau):
            MSDeta[i,j]+=(x_path[0,N,0]-x_path[0,N+tau,0])**2+(x_path[0,N,1]-x_path[0,N+tau,1])**2
        MSDeta[i,j]=MSDeta[i,j]/(time-10)
    
    '''
    x_path=force_calc_stub()    
    #Tau values --------------------------------
    for t in range(0,(int)(tauRes)): 
        tau=t+1
        for N in range(10,time_steps-(tau)):
            #MSDtau[i,t]+=(x_path[0,N,0]-x_path[0,N+tau,0])**2+(x_path[0,N,1]-x_path[0,N+tau,1])**2
            MSDtau[i,t]+=(sqrt((x_path[0,N,0])**2+(x_path[0,N,1])**2) - sqrt((x_path[0,N+tau,0])**2 + (x_path[0,N+tau,1])**2))**2
        MSDtau[i,t]=MSDtau[i,t]/(time_steps-10-(tau-1))
        #print(MSDtau[i,t])
'''
#=======================================================================
# PLOT MSD vs delta tau
#=======================================================================
x_vals=np.zeros((Nspinners,tauRes))
for q in range(0,Nspinners): 
    for u in range(0,(int)(tauRes)):
        x_vals[q,u]=u+1 
plt.scatter(np.log10(x_vals[:,:]),np.log10(MSDtau[:,:]))
#plt.xlim(0,2)
plt.ylabel('(MSD)')
plt.xlabel('(delta tau)')
x=np.log10(x_vals[0,:])
x[x_vals[0,:]==0]=0   
y=np.log10(MSDtau[0,:])
y[MSDtau[0,:]==0]=0 

plt.scatter((x_vals[:,:]),((MSDtau[:,:])))
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
z = np.polyfit(x_vals[0,:], MSDtau[0,:], 1)
p = z[0]*x_vals[0,:] + z[1] 
plt.plot(x_vals[0,:],p[:],"r--")

plt.scatter(x,y)
z = np.polyfit(x, y, 1)
#p = np.log10(z[0])+x_vals[0,:]*z[1]
p = z[0]*x + z[1] 
plt.plot(x,p[:],"r--")
print("MSDtau Fit: y=%.6fx+(%.6f)"%(z[0],z[1]))

plt.savefig("MSDtau_"+ str(jobNum) + "_eta" + str(eta) + ".pdf")
plt.close()

#=======================================================================
# PLOT MSD vs eta
#=======================================================================

x_vals=np.zeros((Nspinners,(int)(1/etaRes)))
for q in range(0,Nspinners): 
    for u in range(0,(int)(1/etaRes)):
        x_vals[q,u]=u*etaRes #increments of 0.1?
plt.scatter((x_vals[:,:]),np.log10(MSDeta[:,:]))
plt.xlim(0,1.1)
plt.ylabel('log(MSD)')
plt.xlabel('eta')
plt.savefig("MSDeta_" + jobNum + ".pdf")
plt.close()
'''
'''
#=======================================================================
# Method call to the vector field calculator
#=======================================================================
x_vf=np.zeros((2*Nres,2*Nres,2)) 
for q in range(-Nres,Nres):
    for u in range(-Nres,Nres):
        x_vf[q,u]=q,u
f_vf=np.zeros((2*Nres,2*Nres,2))
for i in range(-Nres,Nres): #rows
    for j in range(-Nres,Nres): #cols
        f_vf[i,j,0],f_vf[i,j,1]=force_calc(x_vf[i,j]/3)

#=======================================================================
#PLOT THE VECTOR FIELD
#=======================================================================
plot1=plt.figure()
plt.figure(figsize=(10,10))
cm = plt.cm.get_cmap('rainbow')
plt.quiver(x_vf[:,:,0]/3, x_vf[:,:,1]/3, f_vf[:,:,0], f_vf[:,:,1],      
            (np.sqrt(f_vf[:,:,0]**2+f_vf[:,:,1]**2)),                  
            cmap=cm,
            scale=100*omega
            )
lattice1=plt.scatter(x_obst1,y_obst1,s=35,color="blue")
if(basis==2):
    lattice2=plt.scatter(x_obst2,y_obst2,s=35,color="red")
plt.title('Preliminary Vector Field Plot')
plt.xlim(-Nres/3,Nres/3)
plt.ylim(-Nres/3,Nres/3)
plt.savefig("vector_field"+"_eta" + str(eta)+".pdf")
plt.close()

#=======================================================================
# CALL TO RUN THE NUMERICAL MODEL FOR TRAJECTORY
#=======================================================================
'''
#Nspinners=5
path=np.zeros((Nspinners,time_steps,2))
'''
for n in range(Nspinners):
    #print("Spinner: " + str(n))
    x_vec=np.zeros((time_steps,2))
    f_vec=np.zeros((time_steps,2))
    x_vec[0,:]=np.random.randn(2)*10 
    xunit_vec=[1,0]
    yunit_vec=[0,1]
    for i in range(1,time_steps):
        x_vec[i,:]=x_vec[i-1,:]+f_vec[i-1,:]*(dt)
        x_vec[i,0]+=np.sqrt(dt)*noise*np.random.randn()
        x_vec[i,1]+=np.sqrt(dt)*noise*np.random.randn()
        f_vec[i,0],f_vec[i,1]=force_calc(x_vec[i,:]) 
    path[n,:,:]=x_vec

#=======================================================================
# DRAW OUT THE TRAJECTORY IN TIME
#=======================================================================
plt.figure(figsize=((10,10))) 
cm=plt.cm.get_cmap('rainbow')
t=range(time_steps)
plt.quiver(x_vf[:,:,0]/3, x_vf[:,:,1]/3, f_vf[:,:,0], f_vf[:,:,1],      
            (np.sqrt(f_vf[:,:,0]**2+f_vf[:,:,1]**2)),                  
            cmap=cm,
            scale=100*omega
            )
l=plt.scatter(x_obst1,y_obst1,s=30,color="green")
for n in range(Nspinners):
    sc=plt.scatter(path[n,:,0],path[n,:,1], 
                    c=t, 
                    vmin=0, 
                    vmax=time_steps, 
                    s=30, 
                    cmap=cm
                    )
plt.xlim(-20,20)
plt.ylim(-20,20)
#plt.colorbar(sc)
plt.savefig("traj" + str(jobNum) + "_eta" + str(eta) +  ".pdf")
plt.close()
''' 
#=======================================================================
#WRITE OUTPUTS 
#=======================================================================

print("jobNum: " + str(jobNum) + "==========================================================")
print("Run Time : " + str(datetime.now() - startTime))
print("omega: " + str(omega))
print("noise: " + str(noise))
print("Nspinners: " + str(Nspinners))
print("etaRes: " + str(etaRes) + " eta: " + str(eta))
print("gamma:" + str(gamma_t))
print("timesteps: " + str(time_steps))
print("dt: " + str(dt))
print("Nposts: " + str(Nposts))
print("tauRes: " + str(tauRes))
print("Nres: " + str(Nres))
print("alpha: " + str(alpha))
print("Lattice_constant: " + str(lattice_constant))
print("OTHER NOTES: " + "gamma=1-eta")
if(basis==3):
    print("             " + "distorted")
else:
    print("             " + "square")

np.save('MSDtau' + str(jobNum) + '.npy', MSDtau)
np.save('traj' + str(jobNum) + '.npy', path)
np.save('lattice_x.npy', x_obst1)
np.save('lattice_y.npy',y_obst1)
