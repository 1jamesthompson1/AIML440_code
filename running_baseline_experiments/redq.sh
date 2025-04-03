#!/bin/bash

cd /home/thompsjame1/code/REDQ/experiments

task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id
num_seeds=10
num_algos=3
num_envs=9

runs_per_algo=$(($num_seeds * $num_envs))

seed=$(($task_id % $num_seeds))
env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

envs=("HalfCheetah-v5" "Walker2d-v5" "Humanoid-v5" "Ant-v5" "Humanoid-v5" "Swimmer-v5" "Hopper-v5" "InvertedDoublePendulum-v5" "Pusher-v5")

selected_env=${envs[$env_index]}

echo " Running ${selected_env} with seed ${seed}"

echo "==Running train_redq_sac.py=="

poetry run python train_redq_sac.py --seed=${seed} --data_dir=$OUTPUTDIR --env=${selected_env} --epochs=1000
echo "==train_redq_sac.py Complete=="