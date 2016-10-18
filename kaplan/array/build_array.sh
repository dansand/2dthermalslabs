#!/bin/bash
counter=1
for a in 0.15 0.25 0.5
do
   for b in 0.5 1.0 2.0
   do
      for c in 0.75 1.0
      do
        for d in 0.5 1.0 2.0
        do
           qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
           let counter=counter+1
           #if [ "$counter" -gt 1 ]; then break 3; fi;
        done
      done
   done
done
