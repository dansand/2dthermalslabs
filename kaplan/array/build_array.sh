#!/bin/bash
counter=1
for a in 1.0 0.5
do
   for b in 750000.0 250000.0 100000.0
   do
      for c in 150000.0 100000.0
      do
         qsub -v COUNTER=$counter,A=$a,B=$b, C=$c array.pbs
         #echo $counter $counter $a $b $c
         let counter=counter+1
         #if [ "$counter" -gt 1 ]; then break 2; fi;
      done
   done
done
