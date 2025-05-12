#!/bin/bash

# This is going to be used to run the simple baselines algorithms on an invidual grid task.

cd /home/thompsjame1/code/AIML440_code/running_baseline_experiments

echo "==Working Directory=="
pwd
echo "==Listing Files=="
ls

echo "==Understanding which algorithm to run=="
task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id

needed_tasks=(280 281 287 288 289 291 292 293 294 295 296 297 298 299 300 311 312 313 314 315 316 317 318 319 320 342 346 349 351 355 356 357 358)

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