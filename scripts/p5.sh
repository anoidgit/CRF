#!/bin/bash

p_values=("1" "5" "20")

if [[ -f tmp.txt ]];then
	rm tmp.txt
fi

function run_exp() {
	lambda="1e-4"
#	cp ./output/1e-4.txt ./output/p_1.txt
	for p in ${p_values[@]};do
		echo "p = "$p
		$PETSC_DIR/$PETSC_ARCH/bin/mpirun -n 3 ../code_PETSc/seq_train -data ../data/train.txt -tdata ../data/test.txt -lambda $lambda -loss CRF -tol 1e-3 -tao_lmm_vectors $p > "p_"$p".txt"
	done
	mv p_*.txt ./output
}

function plot() {
	plot_setup="set terminal postscript color font ',30'; set style fill pattern border; set xlabel 'CPU time (seconds)'; set yrange[0:*];set ylabel 'object value'; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 0; set output '| ps2pdf - p5_LBFGS.pdf'; "
	for p in ${p_values[@]};do
		echo "p = "$p
		awk '{if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/p_"$p".txt" > tmp$p.txt
	done
	
	echo $plot_setup"plot 'tmp1.txt' using 4:2 w l lw 2 title 'p=1', 'tmp5.txt' using 4:2 w l lw 2 title 'p=5', 'tmp20.txt' using 4:2 w l lw 2 title 'p=20';" | gnuplot
	rm tmp*.txt
}

run_exp
plot
