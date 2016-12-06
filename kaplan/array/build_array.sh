#!/bin/bash
counter=1
for a in 2.0 5.0 10.0 25. 50.
do
    qsub -v COUNTER=$counter,A=$a array.pbs
    #echo $counter $a $b
    let counter=counter+1
    #if [ "$counter" -gt 1 ]; then break 2; fi;
done
