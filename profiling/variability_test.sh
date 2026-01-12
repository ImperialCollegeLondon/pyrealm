#!/bin/bash

#SBATCH -J pyrealm-variability
#SBATCH -A ICCS-SL2-CPU
#SBATCH -p icelake
#SBATCH --output=log/%A_%a.out
#SBATCH --error=log/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=76
#SBATCH --exclusive
#SBATCH --time=02:00:00

module purge
module load rhel8/cclake/base
module load python/3.11.9/gcc/nptrdpll

module=pmodel

n=5

for scaleup in 20 16 14; do
  # ntasks=$(python -c "import math; print(math.ceil($scaleup*1/(3370/1024)))") # scaleup * mem/scaleup / mem/task
  mem=$(python -c "import math; print(math.ceil($scaleup*1))") # scaleup * mem/scaleup
  for i in $(seq $n); do
    echo "RUN $scaleup $i"
    srun -n 1 --${mem}G --cpu-bind=cores ./performance_regression_checking.sh -n HEAD -o HEAD -m $module -s $scaleup &
  done
done

wait
