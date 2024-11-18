#!/bin/bash

while getopts new:old: flag
do
    case "${flag}" in
        n) new_commit=${OPTARG};;
        o) old_commit=${OPTARG};;
        *) echo "Invalid input argument"; exit 1;;
    esac
done

cd ..
git checkout $new_commit

# Remember where we start from
current_repo=`pwd`

#This is the where we want to check the other worktree out to
cmp_repo=$current_repo/../pyrealm_performance_check

# Adding the worktree
echo "Add worktree" $cmp_repo
git worktree add $cmp_repo $old_commit

# Go there and activate poetry environment
cd $cmp_repo
poetry install
#source .venv/bin/activate

# Run the profiling on old commit
echo "Run profiling tests on old commit"
poetry run /usr/bin/time -v pytest -m "profiling" --profile-svg

# Go back into the current repo and run there
cd $current_repo
poetry install
echo "Run profiling tests on new commit"
poetry run /usr/bin/time -v pytest -m "profiling" --profile-svg

# Compare the profiling outputs
cd profiling
python -c "
from pathlib import Path
import simple_benchmarking
import pandas as pd

prof_path_old = Path('$cmp_repo'+'/prof/combined.prof')
print(prof_path_old)
df_old = simple_benchmarking.run_simple_benchmarking(prof_path=prof_path_old)
cumtime_old = (df_old.sum(numeric_only=True)['cumtime'])
print('Old time:', cumtime_old)

prof_path_new = Path('$current_repo'+'/prof/combined.prof')
print(prof_path_new)
df_new = simple_benchmarking.run_simple_benchmarking(prof_path=prof_path_new)
cumtime_new = (df_new.sum(numeric_only=True)['cumtime'])
print('New time:', cumtime_new)

if cumtime_old < cumtime_new:
  print('We got slower. :(')
elif cumtime_new < cumtime_old:
  print('We got quicker! :)')
else:
  print('Times haven\'t changed')
"
cd ..
# Remove the working tree for the comparison commit
echo "Clean up"
git worktree remove --force $cmp_repo
git worktree prune

echo "Done"
