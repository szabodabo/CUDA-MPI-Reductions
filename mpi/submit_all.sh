#!/bin/bash

for NODES in 32 128 512; do
	sbatch -p smallmem --nodes $NODES -t 10 -o ./jobstdout ./ccni_vn.sh
done
