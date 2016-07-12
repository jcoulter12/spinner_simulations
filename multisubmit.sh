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
    w=1
    omega=1
    while [ $w -le 3 ]
    do
		omega=$(($omega*10))
		if [ $w = 1 ]; then
    		ssh aak@18.111.51.247:/Users/aak/Desktop/spinner_simulations/
    	elif [ $w = 2 ]; then 
    		ssh aak@18.111.53.34:/Users/aak/Desktop/spinner_simulations/
    	#else 
    		#ssh aak@18.111.62.233:/Users/aak/Desktop/spinner_simulations/
    	fi
      	mkdir outputs/PDFs-N$noise-W$omega
		cd outputs/PDFs-N$noise-W$omega
		i=0
		while [ $i -le $nJobs ]
		do
			nohup python2.7 /Users/aak/Desktop/spinner_simulations/spinner_sim.py $i $noise $eta $omega >> out.txt &
	   		let "i++"
		done
		exit
		let "w++"
    done
    let "f++"
done
