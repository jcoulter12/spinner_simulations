#!/bin/bash

cd /Users/Donna/Desktop/test/Unslanted_Square/

mkdir PDFs_$(date '+%d-%b-%Y-%k-%M') 
cd PDFs_$(date '+%d-%b-%Y-%k-%M')

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
    	mkdir PDFs-N$noise-W$omega
		cd PDFs-N$noise-W$omega
		omega=$(($omega*10))
		if [ $w = 1 ]; then
			scp aak@18.111.51.247:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
    	elif [ $w = 2 ]; then 
			scp aak@18.111.45.79:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
    	else 
			scp aak@18.111.62.233:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
    	fi
		exit
		let "w++"
    done
    let "f++"
done
