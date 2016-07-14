#!/bin/bash

#cd /Users/Donna/Desktop/test/Unslanted_Square/

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
		echo "cd PDFs-N$noise-W$omega"
		cd PDFs-N$noise-W$omega
		python2.7 /Users/Donna/Desktop/AAK_Group/spinner_simulations/plotting.py >> out.txt
		cd ../
		let "w++"
    done
	let "f++"
done