for a in 0.03 0.015 0.0075 0.00375
do
   for b in 32 64 128
   do
      for c in "Q1/dQ0" "Q2/dpc1"
      do
        docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 8 python Fault_BuoyancyDriven-devel.py 1 pd.fthickness=$a md.RES=$b md.elementType=$c
      done
   done
done
