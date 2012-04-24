set term postscript eps enhanced color

set style line 1 lt 1 lw 3 lc rgb "red" pt 2
set style line 2 lt 1 lw 3 lc rgb "blue" pt 2
set style line 3 lt 1 lw 3 lc rgb "green" pt 2
set style line 4 lt 2 lw 5 lc rgb "red"
set style line 5 lt 2 lw 5 lc rgb "blue"
set style line 6 lt 2 lw 5 lc rgb "green"


set xlabel "Number of MPI Ranks"
set ylabel "Bandwidth (GB/sec)"
set key bottom right

set yrange [0:160]
 
f(x) = 90.8413 
g(x) = 90.7905
h(x) = 90.7969

set output "int.eps"
plot "results/INT_MAX.txt" using 3:4 ls 1 title "BG Max" with linespoints, \
     "results/INT_MIN.txt" using 3:4 ls 2 title "BG Min" with linespoints, \
     "results/INT_SUM.txt" using 3:4 ls 3 title "BG Sum" with linespoints, \
     f(x) ls 4 title "CUDA Sum", \
     g(x) ls 5 title "CUDA Min", \
     h(x) ls 6 title "CUDA Max"

f(x) = 92.7729 
g(x) = 92.6014
h(x) = 92.7552

set output "double.eps"
plot "results/DOUBLE_MAX.txt" using 3:4 ls 1 title "BG Max" with linespoints, \
	"results/DOUBLE_MIN.txt" using 3:4 ls 2 title "BG Min" with linespoints, \
	"results/DOUBLE_SUM.txt" using 3:4 ls 3 title "BG Sum" with linespoints, \
	f(x) ls 4 title "CUDA Sum", \
	g(x) ls 5 title "CUDA Min", \
	h(x) ls 6 title "CUDA Max"

