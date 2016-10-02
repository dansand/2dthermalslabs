for a in "Q1/dQ0" "Q2/dpc1"
do
   for c in `seq 0.0003 0.000005 0.0004`
   do
      docker run -v $PWD:/workspace  -i -t --rm dansand/underworld2-dev mpirun -np 32 python Fault_BuoyancyDriven-devel.py 5 pd.fthickness=0.00775 pd.friction_mu=0.001 pd.friction_C=$c pd.friction_min=0.001 md.RES=128 md.elementType=$a
   done
done
