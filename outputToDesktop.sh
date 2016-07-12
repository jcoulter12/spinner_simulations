#!/bin/bash

cd /Users/Donna/Desktop/test/Unslanted_Square/

mkdir PDFs_$(date '+%d-%b-%Y-%k-%M') 
cd PDFs_$(date '+%d-%b-%Y-%k-%M')

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
    	mkdir PDFs-N$noise-W$omega
		cd PDFs-N$noise-W$omega
		scp aak@18.111.51.247:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
		scp aak@18.111.53.34:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
		scp aak@18.111.120.38:/Users/aak/Desktop/spinner_simulations/outputs/PDFs-N$noise-W$omega/* .
		cd ../
		let "w++"
    done
	let "f++"
done