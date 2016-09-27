#!/bin/bash
counter=11
for a in 128 160 192 256
do
   qsub -v COUNTER=$counter,A=$a array.pbs
   let counter=counter+1
done

