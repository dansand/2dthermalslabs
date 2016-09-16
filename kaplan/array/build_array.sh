#!/bin/bash
counter=1
for x in 0 0.5 1
do
   qsub -v COUNTER=$counter X=$x array.pbs
   let counter=counter+1
done
