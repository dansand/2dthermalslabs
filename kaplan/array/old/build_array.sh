#!/bin/bash
counter=1
for a in 0.2 1.0 2.0
do
   for b in 0.5 1.0 2.0 4.0
   do
      for c in 0.5 1.0 2.0
      do
         for d in 0.1 1.0 10.0
         do 
            qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
            let counter=counter+1
         done
      done
   done
done

