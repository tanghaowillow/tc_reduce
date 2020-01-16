#!/bin/bash
#SBATCH	-p	gpu
#SBATCH	--gres=gpu:1


make
nvprof --print-gpu-trace ./run
#nvprof --metrics achieved_occupancy ./run

make clean