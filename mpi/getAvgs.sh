#!/bin/bash

for DATATYPE in INT DOUBLE; do
	for OP in SUM MIN MAX; do
		OUTFILE=results/${DATATYPE}_${OP}.txt
		echo "" > $OUTFILE
		for NODECOUNT in `cat collected.txt | grep "$DATATYPE $OP" | awk '{ print $3; }' | sort -n | uniq`; do
			MYSUM=`cat collected.txt | grep "$DATATYPE $OP $NODECOUNT" | awk '{ sum += $4 } END { print sum }'`
			MYCOUNT=`cat collected.txt | grep "$DATATYPE $OP $NODECOUNT" | wc -l`
			MYAVG=`echo "scale=5; $MYSUM / $MYCOUNT" | bc`
			echo "$DATATYPE $OP $NODECOUNT $MYAVG" >> $OUTFILE
		done
	done
done
