#!/bin/bash
counter=1
for a in 0.1 0.5 1.0 2.0 5.0
do
   for b in 0.5 1.0 2.0
   do
      for c in 0.75 1.0
      do
        qsub -v COUNTER=$counter,A=$a,B=$b,C=$c array.pbs
        let counter=counter+1
      done
   done
done
