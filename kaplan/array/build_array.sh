#!/bin/bash
counter=1
for a in 0.25 0.75 1.0 1.25 1.5
do
   for b in 0.1 1.0 5.0
   do
     for c in 1.0 10.0 100.
     do
         #qsub -v COUNTER=$counter,A=$a,B=$b,C=$c array.pbs
         echo $counter $a $b $c
         let counter=counter+1
         #if [ "$counter" -gt 1 ]; then break 3; fi;
      done
   done
done
