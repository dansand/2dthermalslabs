#!/bin/bash
counter=1
for a in 0.2 0.4 0.65 0.8 1.0
do
   for b in 0.2 0.4 0.65 0.8 1.0
   do
      #qsub -v COUNTER=$counter,A=$a,B=$b array.pbs
      echo $counter $a $b
      let counter=counter+1
      #if [ "$counter" -gt 1 ]; then break 4; fi;
   done
done
