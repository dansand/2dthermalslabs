for c in `seq 0.0001 0.0005 0.01`
do
   docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 32 python Fault_BuoyancyDriven-devel.py 3 pd.fthickness=0.00775 pd.friction_mu=0.001 pd.friction_C=$c pd.friction_min=0.001 md.RES=128 md.elementType="Q1/dQ0" 
done
