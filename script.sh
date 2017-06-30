#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=00:00:59
# cd ~/textmaker02/
make
time OMP_NUM_THREADS=1 ./a.out
time OMP_NUM_THREADS=2 ./a.out
time OMP_NUM_THREADS=3 ./a.out
time OMP_NUM_THREADS=4 ./a.out
# time OMP_NUM_THREADS=5 ./a.out
# time OMP_NUM_THREADS=6 ./a.out
# time OMP_NUM_THREADS=7 ./a.out
# time OMP_NUM_THREADS=8 ./a.out
# time OMP_NUM_THREADS=9 ./a.out
# time OMP_NUM_THREADS=10 ./a.out
# time OMP_NUM_THREADS=11 ./a.out
# time OMP_NUM_THREADS=12 ./a.out
# time OMP_NUM_THREADS=13 ./a.out
# time OMP_NUM_THREADS=14 ./a.out
# time OMP_NUM_THREADS=15 ./a.out
# time OMP_NUM_THREADS=16 ./a.out
# time OMP_NUM_THREADS=17 ./a.out
# time OMP_NUM_THREADS=18 ./a.out
# time OMP_NUM_THREADS=19 ./a.out
# time OMP_NUM_THREADS=20 ./a.out
# time OMP_NUM_THREADS=21 ./a.out
# time OMP_NUM_THREADS=22 ./a.out
# time OMP_NUM_THREADS=23 ./a.out
# time OMP_NUM_THREADS=24 ./a.out
