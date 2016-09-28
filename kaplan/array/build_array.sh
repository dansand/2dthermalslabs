#!/bin/bash
counter=1
for a in 0.01 0.1 1.0 
do
   for b in 1.0 10.0 100.0 1000.
   do
      for c in 1.0 0.1 0.01
      do
         for d in 1.0 0.1 0.01
         do 
            qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
            let counter=counter+1
         done
      done
   done
done

