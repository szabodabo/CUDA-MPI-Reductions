#!/bin/bash

BGLMPI_MAPPING=TXYZ
INDEX=0

echo 'Starting run...'
mpirun -mode VN \
 -cwd /gpfs/small/PCP2/home/PCP2szab/assignment5 ./reduce \
  > stdout-vn-$SLURM_JOB_ID 2> stderr-vn-$SLURM_JOB_ID

echo 'Job done'
