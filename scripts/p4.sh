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
}

function plot() {
	if [[ -f tmp.txt ]];then
		rm tmp.txt
	fi

	plot_setup="set terminal postscript font ',28'; set style fill pattern border; set xlabel 'CPU time (seconds)'; set yrange[0:*];set ylabel 'object value'; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 0;"
	for n in ${cores[@]};do
		echo $n" cores"
		tail -n 1 "core_"$n".txt" | awk '{print "$n", $4}'>> tmp.txt
		t1=`tail -n 1 "core_1.txt" | awk '{print $4}'`
#		echo $plot_setup" set output '| ps2pdf - p4_scalability.pdf; plot 'tmp.txt' using 1:() w lp notitle;" | gnuplot
	done
	rm tmp.txt
}

run_exp

