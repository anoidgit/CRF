#!/bin/bash

lambdas=("1e-2" "1e-4" "1e-6")

if [[ -f tmp.txt ]];then
	rm tmp.txt
fi

function run_exp() {
	for lambda in ${lambdas[@]};do
		echo "lambda = "$lambda
		$PETSC_DIR/$PETSC_ARCH/bin/mpirun -n 3 ../code_PETSc/seq_train -data ../data/train.txt -tdata ../data/test.txt -lambda $lambda -loss CRF -tol 1e-3 > $lambda.txt
	done
	mv *.txt ./output/
}

function plot() {
	plot_setup="set terminal postscript font ',30'; set style fill pattern border; set xlabel 'CPU time (seconds)'; set yrange[0:*];set ylabel 'object value'; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 0;"
	for lambda in ${lambdas[@]};do
		echo "lambda = "$lambda
		awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  $lambda.txt > tmp.txt
		echo $plot_setup" set output '| ps2pdf - p2_"$lambda".pdf'; plot 'tmp.txt' using 4:2 w l lw 2 notitle;" | gnuplot
	done
	
#	rm tmp.txt
}

#run_exp
plot
