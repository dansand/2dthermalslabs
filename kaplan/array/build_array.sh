#!/bin/bash
counter=1
for a in 0.2 0.5 1.0 2.0 4.0 10.0 20.0
do
   for b in 0.5 0.75 1.0 1.5 2.0 2.5 3.0 4.0
   do
      #qsub -v COUNTER=$counter,A=$a,B=$b array.pbs
      echo $counter $a $b $c $d
      let counter=counter+1
      #if [ "$counter" -gt 1 ]; then break 2; fi;
   done
done
