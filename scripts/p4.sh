#!/bin/bash

cores=("1" "2" "4" "6" "8")

#rm *.txt


function run_exp() {
	lambda="1e-4"
	tol="1e-3"
	
	for n in ${cores[@]};do
		echo "use "$n" cores"
		if [[ $n == "8" ]];then 
			echo $n
			tol="1e-5"
		fi
		echo "$PETSC_DIR/$PETSC_ARCH/bin/mpirun -n $n ../code_PETSc/seq_train -data ../data/train.txt -tdata ../data/test.txt -lambda $lambda -loss CRF -tol $tol > "core_"$n.txt"
		$PETSC_DIR/$PETSC_ARCH/bin/mpirun -n $n ../code_PETSc/seq_train -data ../data/train.txt -tdata ../data/test.txt -lambda $lambda -loss CRF -tol $tol > "core_"$n".txt"
	done
	mv core*.txt ./output/ 
}

function plot() {
	if [[ -f tmp.txt ]];then
		rm tmp.txt
	fi

	plot_setup="set terminal postscript font ',30'; set style fill pattern border; set xlabel 'Number of cores'; set yrange[0:*];set ylabel 'Speedup'; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 0;"

	for n in ${cores[@]};do
		
		f_star=`awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  ./output/core_"$n".txt | tail -n 1 | awk '{print $2}'`
		echo $n" cores, f* = "$f_star
		# get the first line that has f less than 1.01*f_star
		awk '{if($1 ~ /^[0-9]+$/ && NF == 9 && $2 < 1.01*'$f_star') print '$n', $4;}' "./output/core_"$n".txt" | head -n 1 >> tmp.txt
	done
	t1=`awk '{if($1 ~ /^[0-9]+$/ && NF == 9 && $2 < 1.01*'$f_star') print $4;}' ./output/core_1.txt | head -n 1`
	echo $t1
	echo $plot_setup" set output '| ps2pdf - p4_scalability.pdf'; plot 'tmp.txt' using 1:("$t1"/"'$2'") w lp lw 2 ps 2 pt 2 notitle;" | gnuplot
	rm tmp.txt
}

#run_exp
plot
