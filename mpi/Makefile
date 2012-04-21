BG_COMPILER = mpixlc -I.
KR_COMPILER = mpicc -Wall -g -I.

bluegene: clean
	$(BG_COMPILER) reduce.c -o reduce
kratos: 
	$(KR_COMPILER) reduce.c -o reduce
clean:
	rm -f stdout*
	rm -f stderr*
	rm -f core.*
