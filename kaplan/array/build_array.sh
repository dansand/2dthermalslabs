#!/bin/bash
counter=1
for a in "Q1/dQ0" "Q2/DPC1"
do
   for b in "True" "False"
   do
      for c in 128 192 256
      do
        for d in 25 50 75
        do
           qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
           let counter=counter+1
           #if [ "$counter" -gt 1 ]; then break 3; fi;
        done
      done
   done
done
