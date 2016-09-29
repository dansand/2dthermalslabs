for a in 0.015 0.01 0.0075 0.005
do
   for b in 0.00001 0.0001 0.001 0.01
   do
      for c in 0.00001 0.0001 0.001 0.01
      do
         for d in 0.00001 0.0001 0.001 0.01
         do
            or e in 64 128
            do
               for f in "Q1/dQ0" "Q2/dpc1"
                  docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 32 python Fault_BuoyancyDriven-devel.py 2 pd.fthickness=$a pd.friction_mu=$b pd.friction_C=$c pd.friction_min=$d md.RES=$e md.elementType=$f
               done
            done
         done
      done
   done
done
