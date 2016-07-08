#!/bin/bash
nJobs=2
i=0
omega=1
f=0
n=1
w=1
while [ $f -le 3 ]
do
      if [ $f = 0 ]; then
	  noise=0
      else
	  noise=$n
	  n=$(($n*10))
      fi
      export noise
      w=1
      omega=1
      while [ $w -le 3 ]
      do
	  	omega=$(($omega*10))
	  	export omega
	  	echo "cd spinner_simulations/outputs/PDFs-N$noise-W$omega"
	  	i=0
	  	while [ $i -le $nJobs ];
	  	do
	   		jobNum=$i
	    	export jobNum
	      	eta=$(echo "scale=1;0.1+$i*0.4" | bc)
	      	#echo $eta
	      	export eta
	      	#bsub -R "pool>5000" -M 3000000 -q 1nd -J merge_job_${eta} < /afs/cern.ch/work/j/jcoulter/WORK/CMSSW_5_3_20/src/tests/submit.sh
	      	bash submit.sh
	      	let "i++"
	  	done
	  echo "cd ../../../"
	  let "w++"
      done
      let "f++"
done