#!/bin/bash

lambda="1e-4"

if [[ -f tmp.txt ]];then
	rm tmp.txt
fi

plot_setup="set terminal postscript color font ',30'; set style fill pattern border; set xlabel 'Effective number of passes'; set yrange [0:*]; set ytics 0.2; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 0;"

awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/lbfgs_"$lambda.txt > tmp1.txt
awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/sgd_"$lambda.txt > tmp2.txt
awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/sag_nus"$lambda.txt > tmp3.txt

echo $plot_setup" set ylabel 'Letter-wise error'; set output '| ps2pdf - t3_letter_err.pdf'; plot 'tmp1.txt' using 4:("'$8'"/100.0)  w l lw 4 dt 6 lc rgb 'red' title 'LBFGS';, 'tmp2.txt' using 4:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'black' title 'SGD', 'tmp3.txt' using 4:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'green' title 'SAG-NUS'" | gnuplot
echo $plot_setup" set ylabel 'Word-wise error'; set output '| ps2pdf - t3_word_err.pdf'; plot 'tmp1.txt' using 4:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'red' title 'LBFGS', 'tmp2.txt' using 4:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'black' title 'SGD', 'tmp3.txt' using 4:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'green' title 'SAG-NUS';" | gnuplot

rm tmp*.txt
