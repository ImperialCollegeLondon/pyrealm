#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "No input arguments, comparing HEAD to origin/develop"
    new_commit=HEAD
    old_commit=origin/develop
else
    while getopts n:o:a:m:s: flag
    do
        case "${flag}" in
            n) new_commit=${OPTARG};;
            o) old_commit=${OPTARG};;
            a) advanced=true;;
            m) module=${OPTARG};;
            s) scaleup=${OPTARG};;
            *) echo "Invalid input argument"; exit 1;;
        esac
    done
fi

cd $(git rev-parse --show-toplevel)

# Remember where we start from
current_repo=`pwd`

# Perform the profiling. First on the old commit, then the new commit
for version in "old" "new"; do
    repo=$current_repo/../pyrealm_performance_check_$version
    commit_var=${version}_commit
    commit=${!commit_var}

    # Adding the worktree
    echo "Add worktree" $repo
    trap "git worktree remove --force $repo" EXIT
    git worktree add $repo $commit

    # Go there and activate poetry environment
    cd $repo
    unset VIRTUAL_ENV
    export POETRY_VIRTUALENVS_IN_PROJECT=1
    poetry install > /dev/null

    # Run the profiling
    echo "Run profiling tests on $version commit"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then #Linux
        poetry run /usr/bin/time -v pytest -m "profiling" --profile --$module-profile-scaleup $scaleup tests/profiling/$module
    elif [[ "$OSTYPE" == "darwin"* ]]; then #Mac OS
         poetry run /usr/bin/time -l pytest -m "profiling" --profile --$module-profile-scaleup $scaleup  tests/profiling/$module
    fi
    if [ "$?" != "0" ]; then
        echo "Profiling the current code went wrong."
        exit 1
    fi

    # Perform the advanced function-by-function analysis
    if [ "$advanced" = true ]; then
        if [ "$version" = "old" ]; then
            rm $current_repo/profiling/profiling-database.csv
            rm $current_repo/prof/performance-plot.png
            # Initialise the database
            echo "Advanced regression test: Initialising database"
            poetry run python $current_repo/profiling/run_benchmarking.py \
                prof/combined.prof \
                $current_repo/profiling/profiling-database.csv \
                $current_repo/profiling/benchmark-fails.csv \
                $commit \
            ||
            advanced_failed=true
        else
            # Compare the new times against the database
            echo "Advanced regression test: Comparing new profiling results"
            poetry run python $current_repo/profiling/run_benchmarking.py \
                prof/combined.prof \
                $current_repo/profiling/profiling-database.csv \
                $current_repo/profiling/benchmark-fails.csv \
                $commit \
                --plot-path $current_repo/prof/performance-plot.png \
            ||
            advanced_failed=true
        fi
    fi

    # Copy output and go back to the current repo
    cp "prof/combined.prof" "$current_repo/prof/combined-$version.prof"
    cd $current_repo

    # Remove the worktree
    git worktree remove --force $repo
    git worktree prune
    trap - EXIT
done

# Compare the profiling outputs
cd profiling
poetry run python -c "
from pathlib import Path
import simple_benchmarking
import pandas as pd
import sys

prof_path_old = Path('$current_repo'+'/prof/combined-old.prof')
print(prof_path_old)
df_old = simple_benchmarking.run_simple_benchmarking(prof_path=prof_path_old)
cumtime_old = (df_old.sum(numeric_only=True)['cumtime'])
print('Old time:', cumtime_old)
with open('$module-variability.txt', 'a') as f:
  f.write(f'$scaleup {cumtime_old}\n')

prof_path_new = Path('$current_repo'+'/prof/combined-new.prof')
print(prof_path_new)
df_new = simple_benchmarking.run_simple_benchmarking(prof_path=prof_path_new)
cumtime_new = (df_new.sum(numeric_only=True)['cumtime'])
print('New time:', cumtime_new)
with open('$module-variability.txt', 'a') as f:
  f.write(f'$scaleup {cumtime_new}\n')

if cumtime_old < 0.95*cumtime_new:
  print('We got slower. :(')
  sys.exit(1)
elif cumtime_new < 0.95*cumtime_old:
  print('We got quicker! :)')
else:
  print('Times haven\'t changed')
"
benchmarking_out="$?"

# Report the results of the advanced regression test
if [ "$advanced" = true ]; then
    echo
    if [ "$advanced_failed" = true ]; then
        echo "Advanced regression test failed."
    else
        echo "Advanced regression test succeeded."
    fi
    if [ -f $current_repo/prof/performance-plot.png ]; then
      echo "View the results in: prof/performance-plot.png"
    fi
    echo
fi

# Remove the profiling outputs
rm "$current_repo/prof/combined-old.prof"
rm "$current_repo/prof/combined-new.prof"

if [ $benchmarking_out != "0" ]; then
    echo "The new code is more than 5% slower than the old one."
    exit 1
fi

echo "No significant performance regression detected."
