#!/bin/bash

# This is a general purpose script that is designed to be a run a cluster system and it setups an experiment on one of my on creations.
# This script needs to be passed a task id as an argument as well as the algorithm to run as another arguement

cd ~/code/AIML440_code/my-algorithm-implementation/

task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id
num_seeds=10
num_algos=3
num_envs=9

runs_per_algo=$(($num_seeds * $num_envs))

seed=$(($task_id % $num_seeds))
env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

envs=("HalfCheetah-v5" "Walker2d-v5" "Humanoid-v5" "Ant-v5" "HumanoidStandup-v5" "Swimmer-v5" "Hopper-v5" "InvertedDoublePendulum-v5" "Pusher-v5")

selected_env=${envs[$env_index]}

echo " Running ${selected_env} with seed ${seed}"

training_script=$2/train.py

echo "==Running ${training_script}=="

poetry run python ${training_script} --seed=${seed} --data-dir=$OUTPUTDIR --env=${selected_env}
echo "==${training_script} Complete=="