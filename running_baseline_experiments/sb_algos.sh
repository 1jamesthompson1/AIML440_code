#!/bin/bash

# This is going to be used to run the simple baselines algorithms on an invidual grid task.

cd ~/code/AIML440_code/running_baseline_experiments

echo "==Working Directory=="
pwd
echo "==Listing Files=="
ls

echo "==Understanding which algorithm to run=="
task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id

needed_tasks=(279 286 290 298 310 341 345 348 350 354)

task_id=${needed_tasks[$task_id]}

num_seeds=10
num_algos=4
num_envs=9

runs_per_algo=$(($num_seeds * $num_envs))

seed=$(($task_id % $num_seeds))
algo_index=$(( ($task_id / $runs_per_algo) % $num_algos))
env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

algorithms=("ppo" "sac" "td3" "crossq")
selected_algo=${algorithms[$algo_index]}

echo " Running ${selected_algo} in env ${env_index} with seed ${seed}"

echo "==Running Script=="

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=''
poetry run python train.py --seed=${seed} --output_dir=$OUTPUTDIR --algo=${selected_algo} --env_index=${env_index}

echo "==Script Complete=="