#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3988
#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=gregor.steiner@warwick.ac.uk

module purge
module load Julia/1.8.2-linux-x86_64
module load GCC/11.3.0 OpenMPI/4.1.4
module load GCC/11.3.0 OpenMPI/4.1.4 R/4.2.1

# Make sure our extra packages are available
export JULIA_DEPOT_PATH=~/julia

srun julia simulation_Kang2016.jl
