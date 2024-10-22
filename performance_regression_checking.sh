#!/bin/bash

# Remember where we start from
current_repo=`pwd`

#This is the where we want to check the other worktree out to
cmp_repo=../pyrealm_performance_check

# Adding the worktree
echo "Add worktree" $cmp_repo
git worktree add $cmp_repo HEAD~1

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
poetry run /usr/bin/time -v pytest -m "profiling" --profile-svg

# Compare the profiling outputs



# Remove the working tree for the comparison commit
echo "Clean up"
git worktree remove --force $cmp_repo
git worktree prune

echo "Done"
