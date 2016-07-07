#!/bin/bash

echo "nohup python spinner_sim.py"

cd outputs/
mkdir PDFs_$(date '+%d-%b-%Y-%k-%M') 
cd PDFs_$(date '+%d-%b-%Y-%k-%M')

#nJobs=1
#i=0
#echo "Number of Jobs: $nJobs" > out.txt
echo "=================================" >> out.txt
#chmod +w out.txt
#while [ $i -le $nJobs ];
#do
#	nohup python $HOME/Desktop/AAK_Group/spinner_simulations/spinner_sim.py $i >> out.txt
#  	let "i++"
#done

noise=0.0s
nohup & python $HOME/Desktop/AAK_Group/spinner_simulations/spinner_sim.py 0 $noise 0.1 >> out.txt
nohup & python $HOME/Desktop/AAK_Group/spinner_simulations/spinner_sim.py 1 $noise 0.5 >> out.txt
nohup & python $HOME/Desktop/AAK_Group/spinner_simulations/spinner_sim.py 2 $noise 0.9 >> out.txt

cd ../

echo "Done!"