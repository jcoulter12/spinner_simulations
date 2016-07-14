#!/bin/bash
nJobs=2
i=0
omega=1
f=0
n=1
w=1
while [ $f -le 2 ]
do
    if [ $f = 0 ]; then
	  noise=0
    else
	  noise=$n
	  n=$(($n*10))
    fi
    #export noise
    w=1
    omega=1
    while [ $w -le 3 ]
    do
		omega=$(($omega*10))
		#export omega
      	echo "mkdir outputs/PDFs-N$noise-W$omega"
		echo "cd outputs/PDFs-N$noise-W$omega"
		i=0
		eta=0
		echo "nohup python2.7 /Users/aak/Desktop/spinner_simulations/spinner_sim.py $i $noise $eta $omega >> out.txt &"
		while [ $i -le $nJobs ];
		do
	   		jobNum=$(($i+1))
	   		#export jobNum
	   		eta=$(echo "scale=1;0.1+$i*0.4" | bc)
	  		#export eta
	   		echo "nohup python2.7 /Users/aak/Desktop/spinner_simulations/spinner_sim.py $i $noise $eta $omega >> out.txt &"
	   		let "i++"
		done
		echo "cd ../../"
		let "w++"
    done
    let "f++"
done
