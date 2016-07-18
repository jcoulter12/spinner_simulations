#!/bin/bash
nJobs=4
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
	w=0
	while [ $w -le 3 ]
	do	
		if [ $w = 0 ]; then
			omega=0
		else
			omega=$((10**$w))
		fi
		#echo "mkdir outputs/PDFs-N$noise-W$omega"
		echo "cd outputs/PDFs-N$noise-W$omega"
		echo "echo =============================== > out.txt"
		i=0
		eta=0
		# this is just here to get the eta = 0 case
		#echo "nohup python2.7 /Users/aak/Desktop/spinner_simulations/spinner_sim.py $i $noise $eta $omega >> out.txt &"
		while [ $i -le $nJobs ];
		do
			#jobNum=$(($i+1))
			#eta=$(echo "scale=1;0.1+$i*0.4" | bc)
			eta=$(echo "scale=1;$i*0.25" | bc)
			echo "nohup python2.7 /Users/aak/Desktop/spinner_simulations/spinner_sim.py $i $noise $eta $omega >> out.txt &"
			let "i++"
		done
		echo "cd ../../"
		let "w++"
	done
	let "f++"
done
