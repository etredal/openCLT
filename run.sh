#!/bin/bash


# Job details
TIME=04:00:00  # HH:MM:SS (default: 04:00, max: 240:00)
NUM_GPUS=2  # GPUs per node
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=8024  # RAM for each core (default: 1024)
OUTFILE=test.out  # default: lsf.oJOBID
script=test.sh


# Load modules
# module load python_gpu/3.6.1


# Submit job
sbatch –time=$TIME \
     -n $NUM_CPUS \
     --mem-per-cpu=$CPU_RAM -G $NUM_GPUS  \
     –gpus=nvidia_a100-pcie-40gb \
     -o $OUTFILE \
     ./$script
