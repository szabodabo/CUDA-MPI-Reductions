set term postscript eps enhanced color

set xlabel "Number of MPI Ranks"
set ylabel "Bandwidth (GB/sec)"

set output "int.eps"
plot "results/INT_MAX.txt" using 3:4 title "Int Max" with linespoints, \
     "results/INT_MIN.txt" using 3:4 title "Int Min" with linespoints, \
     "results/INT_SUM.txt" using 3:4 title "Int Sum" with linespoints

set output "double.eps"
plot "results/DOUBLE_MAX.txt" using 3:4 title "Double Max" with linespoints, \
		 "results/DOUBLE_MIN.txt" using 3:4 title "Double Min" with linespoints, \
		 "results/DOUBLE_SUM.txt" using 3:4 title "Double Sum" with linespoints

