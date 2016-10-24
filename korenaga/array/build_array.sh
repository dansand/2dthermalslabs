#!/bin/bash
counter=1


for a in 'None' 'Iso' 'Trans'
do
   for b in 1.0 2.0 4.0 8.0
   do
      echo $a $b $counter
      #qsub -v COUNTER=$counter,D=$d,A=$a,B=$b,C=$c array.pbs
      let counter=counter+1
      #if [ "$counter" -gt 1 ]; then break 2; fi; #use this line to test limited set
   done
done
