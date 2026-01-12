#!/bin/bash

module=pmodel
scaleup=4

n=5

# > $module-variability.txt

for i in $(seq $n); do
  echo ./performance_regression_checking.sh -n HEAD -o HEAD -m $module -s $scaleup
  ./performance_regression_checking.sh -n HEAD -o HEAD -m $module -s $scaleup
done
