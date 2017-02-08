for l in `seq 0.0005 0.0005 0.0205`
do
   docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 32 python Fault_BuoyancyDriven-devel.py 12 pd.fthickness=$l pd.friction_C=0.0002 md.RES=128 md.elementType="Q2/dpc1" pd.deltaViscosity=20 pd.orientation=30
done 
