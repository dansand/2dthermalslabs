#!/bin/bash
counter=1

for d in 1.0 2.0 10.0
do
   for a in 1.0 1.3 1.8
   do
      for b in 0.5 0.1
      do
         for c in 1.0 1.4
         do
            qsub -v COUNTER=$counter,D=$d,A=$a,B=$b,C=$c array.pbs
            let counter=counter+1
         done
      done
   done
done
