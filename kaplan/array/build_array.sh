#!/bin/bash
counter=1
for a in 0.3 0.6 0.9
do
   for b in 0.5 1.0
   do
      for c in 0.5 1.0 1.5
      do
         for d in 0.01 0.1 1.0
         do
            #qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
            echo $counter $a $b $c $d
            let counter=counter+1
            #if [ "$counter" -gt 1 ]; then break 4; fi;
         done
      done
   done
done
