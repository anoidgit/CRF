#!/bin/bash

lambda="1e-4"

if [[ -f tmp.txt ]];then
	rm tmp.txt
fi

plot_setup="set terminal postscript color font ',30'; set style fill pattern border; set xlabel 'CPU time (seconds)'; set yrange [0:*]; set ytics 0.2; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 0;"

awk '{ if($1 ~ /^[0-9]+$/ && NF == 9) print $0}'  "./output/"$lambda.txt > tmp.txt

echo $plot_setup" set ylabel 'Letter-wise error'; set output '| ps2pdf - p3_letter_err.pdf'; plot 'tmp.txt' using 4:("'$6'"/100.0) w l lw 2 lc rgb 'black' title 'Training', 'tmp.txt' using 4:("'$8'"/100.0)  w l lw 4 dt 6 lc rgb 'red' title 'Testing';" | gnuplot
echo $plot_setup" set ylabel 'Word-wise error'; set output '| ps2pdf - p3_word_err.pdf'; plot 'tmp.txt' using 4:("'$7'"/100.0) w l lw 2 lc rgb 'black' title 'Training', 'tmp.txt' using 4:("'$9'"/100.0) w l lw 4 dt 6 lc rgb 'red' title 'Testing';" | gnuplot

