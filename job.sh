#!/bin/sh

# create environment via env.yml file if not exists already
if conda info --envs | grep -q DynTrap_env; then echo "===== DynTrap_env already exists"; else conda env create -n DynTrap_env --file env.yml; fi

# activate the environment to run code in it
conda activate DynTrap_env


timesteps=200000
timestepsMD=100
# add NSW steps to top of INCAR file
sed -i "1s/^/NSW=$timestepsMD\n/" INCAR
t=0
while [ $t -le $timesteps ] 
do
	echo '========   Processing time = '$t
	mpirun -np $SLURM_NTASKS $VASP
	# main.py returns the timestep where the polaron jumps and updates the POSCAR for that timesteps
	t=$(($t+$(python polaron_dynamic.py)))
	mv OUTCAR OUTCARS/OUTCAR_$t
done
python create_OUTCAR.py

conda deactivate