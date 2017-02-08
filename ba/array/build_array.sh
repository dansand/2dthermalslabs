#!/bin/bash
counter=1
for a in 0.05 0.1 0.25
do
   for b in 0.2 1.0 5.0
   do
      for c in 6000.0 8000.0 10000.0
      do
         for d in 75000.0 100000.0 150000.0
         do
            #qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
            echo $counter $a $b $c $d
            let counter=counter+1
            #if [ "$counter" -gt 1 ]; then break 4; fi;
         done
      done
   done
done
