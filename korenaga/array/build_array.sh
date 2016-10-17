#!/bin/bash
counter=1
for a in 0.8 1.0 1.5 2.0 3.0
do
   for b in 0.5 0.1 1.5
   do
      for c in 1.0 1.3 1.6
      do
        qsub -v COUNTER=$counter,A=$a,B=$b,C=$c array.pbs
        let counter=counter+1
      done
   done
done
