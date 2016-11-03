#!/bin/bash

#lambdas=("1e-2" "1e-4" "1e-6")
lambdas=("1e-6")

if [[ -f tmp.txt ]];then
	rm tmp.txt
fi

function run_exp() {
	cd ../code_Torch
	for lambda in ${lambdas[@]};do
		echo "lambda = "$lambda
#		th ./train_CRF.lua -lambda $lambda -optim lbfgs > "lbfgs_"$lambda".txt"
#		th ./train_CRF.lua -lambda $lambda -optim sgd > "sgd_"$lambda".txt"
		th ./train_CRF.lua -lambda $lambda -optim sag > "sag_nus_"$lambda".txt"
	done
	cd -
#	mv ../code_Torch/*.txt ./output/
}

function plot() {
	plot_setup="set terminal postscript color font ',30'; set style fill pattern border; set xlabel 'CPU time (seconds)'; set yrange[0:*]; set xtics 200; set ylabel 'Training object value'; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 2;"
	for lambda in ${lambdas[@]};do
		echo "lambda = "$lambda
		awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/lbfgs_"$lambda.txt > tmp1.txt
		awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/sgd_"$lambda.txt > tmp2.txt
		awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/sag_nus_"$lambda.txt > tmp3.txt
		echo $plot_setup" set output '| ps2pdf - t2_"$lambda".pdf'; plot 'tmp1.txt' using 4:2 w l lw 2 title 'LBFGS', 'tmp2.txt' using 4:2 w l lw 2 title 'SGD', 'tmp3.txt' using 4:2 w l lw 2 title 'SAG-BUS';" | gnuplot
	done
	
	rm tmp*.txt
}

run_exp
#plot
