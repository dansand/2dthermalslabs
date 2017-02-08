#!/bin/bash
counter=1
for a in 128 160 192 256 320
do
   for b in "Q1/dQ0" "Q2/DPC1"
   do
      qsub -v COUNTER=$counter,A=$a,B=$b array.pbs
      #echo $counter $a $b
      let counter=counter+1
      if [ "$counter" -gt 1 ]; then break 2; fi;
   done
done
