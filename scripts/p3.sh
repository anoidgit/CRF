#!/bin/bash

lambda="1e-4"

if [[ -f tmp.txt ]];then
	rm tmp.txt
fi

plot_setup="set terminal postscript font ',28'; set style fill pattern border; set xlabel 'CPU time (seconds)'; set yrange [0:*]; set lmargin 5; set tmargin 0; set bmargin 1;set rmargin 0;"

awk '$1 ~ /^[0-9]+$/ {print $0}'  $lambda.txt > tmp.txt

echo $plot_setup" set ylabel 'Letter-wise error'; set output '| ps2pdf - p3_letter_err.pdf; plot 'tmp.txt' using 6:xtic(4) title 'Training', using 8:xtic(1) title '$Testing';" | gnuplot
echo $plot_setup" set ylabel 'Word-wise error'; set output '| ps2pdf - p3_word_err.pdf; plot 'tmp.txt' using 7:xtic(4) title 'Training', using 9:xtic(1) title '$Testing';" | gnuplot

