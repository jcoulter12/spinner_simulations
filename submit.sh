#!/bin/bash

echo "nohup python spinner_sim.py"

cd outputs/
#mkdir PDFs_$(date '+%d-%b-%Y-%k-%M') 
#cd PDFs_$(date '+%d-%b-%Y-%k-%M')


#nJobs=1
#i=0
#echo "Number of Jobs: $nJobs" > out.txt
#chmod +w out.txt
#while [ $i -le $nJobs ];
#do
#	nohup python $HOME/Desktop/AAK_Group/spinner_simulations/spinner_sim.py $i >> out.txt
#  	let "i++"
#done

eta=0.5
omega=100
noise=0
mkdir PDFs-N$noise-E$eta-W$omega
cd PDFs-N$noise-E$eta-W$omega

echo "=================================" >> out.txt

nohup python $HOME/Desktop/AAK_Group/spinner_simulations/spinner_sim.py 0 $noise $eta >> out.txt &
#nice -19 nohup python $HOME/Desktop/AAK_Group/spinner_simulations/spinner_sim.py 1 $noise 0.5 >> out.txt &
#nice -19 nohup python $HOME/Desktop/AAK_Group/spinner_simulations/spinner_sim.py 2 $noise 0.9 >> out.txt &

cd ../

echo "Done!"