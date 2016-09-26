#!/bin/bash
counter=1
for a in 64 92 128 160
do     
   qsub -v COUNTER=$counter,A=$a,B=$b,C=$c,D=$d array.pbs
   let counter=counter+1
done


