#!/bin/bash
counter=1
for a in 0.1 0.5 1.0 2.0 10. 
do
   for b in 0.8 1.2 1.6
   do
      for c in 0.5 1.0 2.0
      do
         for d in 128 192
         do 
            qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
            let counter=counter+1
         done
      done
   done
done

