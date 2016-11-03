#!/bin/bash

lambda="1e-4"

if [[ -f tmp.txt ]];then
	rm tmp.txt
fi

function run_exp() {
	cd ../code_Torch
	for lambda in ${lambdas[@]};do
		echo "lambda = "$lambda
		th ./train_CRF.lua -lambda $lambda -optim sgd > "sgd_all_"$lambda".txt"
#		th ./train_CRF.lua -lambda $lambda -optim sag > "sag_nus_all_"$lambda".txt"
	done
	cd -
	mv ../code_Torch/*.txt ./output/
}

function plot() {

	plot_setup="set terminal postscript color font ',30'; set style fill pattern border; set xlabel 'Effective number of passes'; set yrange [0:*]; set ytics 0.2; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 0;"
	
	awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/"$lambda.txt > tmp1.txt		# copy results from PETSc 
	awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/sgd_all_"$lambda.txt > tmp2.txt
	awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/sag_nus_all_"$lambda.txt > tmp3.txt
	
	echo $plot_setup" set ylabel 'Letter-wise error'; set output '| ps2pdf - t3_letter_err.pdf'; plot 'tmp1.txt' using 5:("'$8'"/100.0)  w l lw 4 dt 6 lc rgb 'red' title 'LBFGS';, 'tmp2.txt' using 5:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'black' title 'SGD', 'tmp3.txt' using 5:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'green' title 'SAG-NUS'" | gnuplot
	echo $plot_setup" set ylabel 'Word-wise error'; set output '| ps2pdf - t3_word_err.pdf'; plot 'tmp1.txt' using 5:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'red' title 'LBFGS', 'tmp2.txt' using 5:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'black' title 'SGD', 'tmp3.txt' using 5:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'green' title 'SAG-NUS';" | gnuplot
	
	rm tmp*.txt
}
