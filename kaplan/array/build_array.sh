#!/bin/bash
counter=1
for a in 64 92 128 160 192
do
   for b in True False
   do
      qsub -v COUNTER=$counter,A=$a,B=$b array.pbs
      let counter=counter+1
   done
done

