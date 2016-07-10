import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import os
import math
from sys import argv
from time import strftime

#script,jobNum = argv
jobNum=0
#MSDtau=np.zeros(len(np.open('MSDtau0.npy'))

#MSDeta[i,:]=np.load('MSDeta'+str(i)+'.npy')
#print(MSDeta[i])
MSDtau=np.load('MSDtau'+str(jobNum)+'.npy')
x_obst1=np.load('lattice_x.npy')
y_obst1=np.load('lattice_y.npy')
path=np.load('traj' + str(jobNum) + '.npy')

#=======================================================================
# PLOT MSD vs delta tau
#=======================================================================
x_vals=np.zeros(len(MSDtau))
for u in range(0,len(MSDtau)):
    x_vals[u]=u+1 
plt.scatter(np.log10(x_vals[:]),np.log10(MSDtau[:]))
plt.ylabel('(MSD)')
plt.xlabel('(delta tau)')
x=np.log10(x_vals[:])
x[x_vals[:]==0]=0   
y=np.log10(MSDtau[:])
y[MSDtau[:]==0]=0 

plt.scatter(x,y)
z = np.polyfit(x, y, 1)
#p = np.log10(z[0])+x_vals[:]*z[1]
p = z[0]*x + z[1] 
plt.plot(x,p[:],"r--")
print("MSDtau Fit: y=%.6fx+(%.6f)"%(z[0],z[1]))

plt.savefig("MSDtau_"+ str(jobNum) + ".pdf")
plt.close()

#=======================================================================
# DRAW OUT THE TRAJECTORY IN TIME
#=======================================================================
time_steps=len(path[0])
Nspinners=len(path)
plt.figure(figsize=((10,10))) 
cm=plt.cm.get_cmap('rainbow')
t=range(time_steps)
#plt.quiver(x_vf[:,:,0]/3, x_vf[:,:,1]/3, f_vf[:,:,0], f_vf[:,:,1],      
#            (np.sqrt(f_vf[:,:,0]**2+f_vf[:,:,1]**2)),                  
#            cmap=cm,
#            scale=10000
#            )
l=plt.scatter(x_obst1,y_obst1,s=30,color="green")
for n in range(Nspinners):
    sc=plt.scatter(path[n,:,0],path[n,:,1], 
                    c=t, 
                    vmin=0, 
                    vmax=time_steps, 
                    s=30, 
                    cmap=cm
                    )
plt.xlim(-30,30)
plt.ylim(-30,30)
plt.savefig("traj" + str(jobNum) + ".pdf")
plt.close()

print("done!")