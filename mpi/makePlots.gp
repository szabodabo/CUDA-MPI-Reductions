set term postscript eps enhanced color

set xlabel "Number of MPI Ranks"
set ylabel "Bandwidth (GB/sec)"
set key bottom right
 
f(x) = 90.8413 
g(x) = 90.7905
h(x) = 90.7969

set output "int.eps"
plot "results/INT_MAX.txt" using 3:4 title "BG Max" with linespoints, \
     "results/INT_MIN.txt" using 3:4 title "BG Min" with linespoints, \
     "results/INT_SUM.txt" using 3:4 title "BG Sum" with linespoints, \
     f(x) title "CUDA Sum", \
     g(x) title "CUDA Min", \
     h(x) title "CUDA Max"

f(x) = 92.7729 
g(x) = 92.6014
h(x) = 92.7552

set output "double.eps"
plot "results/DOUBLE_MAX.txt" using 3:4 title "Double Max" with linespoints, \
	"results/DOUBLE_MIN.txt" using 3:4 title "Double Min" with linespoints, \
	"results/DOUBLE_SUM.txt" using 3:4 title "Double Sum" with linespoints, \
	f(x) title "CUDA Sum", \
	g(x) title "CUDA Min", \
	h(x) title "CUDA Max"

