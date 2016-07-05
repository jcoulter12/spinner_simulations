#!/bin/bash

echo "nohup python /$HOME/Desktop/AAK_Group/spinner_sim.py"

cd outputs/

mkdir PDFs_$(date '+%d-%b-%Y-%k-%M') 

cd PDFs_$(date '+%d-%b-%Y-%k-%M')

#nohup python $HOME/Desktop/AAK_Group/test.py >> out.txt

nJobs=10
i=0
echo "Number of Jobs: $nJobs" > out.txt
echo "=================================" >> out.txt
chmod +w out.txt
while [ $i -le $nJobs ];
do
	nohup python $HOME/Desktop/AAK_Group/spinner_sim.py $i >> out.txt
  	let "i++"
done

#cd ../

echo "Done!"