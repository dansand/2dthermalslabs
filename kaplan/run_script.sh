for e in 5.0 10. 20. 50.
do
   for o in 15.0 30.0
   do
      docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 32 python Fault_BuoyancyDriven-devel.py 9 pd.fthickness=0.0075 pd.friction_C=0.0002 md.RES=128 md.elementType="Q2/dpc1" pd.deltaViscosity=$e pd.orientation=$o
   done
done
