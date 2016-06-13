#!/bin/bash
for i in  10 7 4 1
do
    mpirun -np 16 python A1.py $i
done
