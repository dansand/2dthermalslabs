for a in "Q1/dQ0" "Q2/dpc1"
do
   for for l in `seq 0.0005 0.0005 0.0205`
   do
      docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 32 python Fault_BuoyancyDriven-devel.py 7 pd.fthickness=$l pd.friction_mu=0.001 pd.friction_C=0.0002 pd.friction_min=0.001 md.RES=128 md.elementType=$a
   done
done
