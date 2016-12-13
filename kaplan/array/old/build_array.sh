#!/bin/bash
counter=1
for a in 0.25 0.5 1.0 2.0 4.0
do
   for b in 0.25 0.5 1.0 2.0 4.0
   do
      for c in 0.1 0.25 0.5 1.0 2.0
      do
         for d in 0.1 0.25 0.5 1.0
         do
            #qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
            echo $counter $a $b $c $d
            let counter=counter+1
            if [ "$counter" -gt 1 ]; then break 4; fi;
         done
      done
   done
done
