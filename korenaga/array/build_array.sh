#!/bin/bash
counter=1


for a in 'None' 'Iso' 'Trans'
do
   for b in 1. 2. 5. 10
   do
      for c in 0.5 0.1 0.05 0.01
      do
         qsub -v COUNTER=$counter,D=$d,A=$a,B=$b,C=$c array.pbs
         let counter=counter+1
         if [ "$counter" -gt 1 ]; then break 3; fi; #use this line to test limited set
      done
   done
done
