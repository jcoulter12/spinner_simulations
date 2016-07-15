#!/bin/bash

cd /Users/Donna/Desktop/test/Unslanted_Square/

#mkdir PDFs_$(date '+%d-%b-%Y-%k-%M') 
#cd PDFs_$(date '+%d-%b-%Y-%k-%M')
mkdir PDFs_please1
cd PDFs_please1

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
    w=1
    omega=1
    while [ $w -le 3 ]
    do
		omega=$(($omega*10))
		mkdir PDFs-N$noise-W$omega
		cd PDFs-N$noise-W$omega
		if [ $f = 0 ]; then
			scp aak@18.111.51.247:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
    	elif [ $f = 1 ]; then 
			scp aak@18.111.97.37:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
    	else 
			scp aak@18.111.62.233:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
    	fi
		let "w++"
		cd ../
    done
    let "f++"
done
