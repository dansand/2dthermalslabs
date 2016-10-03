for e in 1.0 10. 100. 1000.
do
   for o in 30.0 45.0 60.0
   do
      docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 32 python Fault_BuoyancyDriven-devel.py 8 pd.fthickness=0.0075 pd.friction_C=0.0002 md.RES=128 md.elementType="Q2/dpc1" pd.deltaViscosity=$e pd.orientation=$o
   done
done
