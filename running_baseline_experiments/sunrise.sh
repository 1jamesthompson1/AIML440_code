#!/bin/bash

cd ~/code/sunrise

task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id
num_seeds=10
num_envs=9

runs_per_algo=$(($num_seeds * $num_envs))

seed=$(($task_id % $num_seeds))
env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

envs=("HalfCheetah-v5" "Walker2d-v5" "Humanoid-v5" "Ant-v5" "HumanoidStandup-v5" "Swimmer-v5" "Hopper-v5" "InvertedDoublePendulum-v5" "Pusher-v5")

selected_env=${envs[$env_index]}

# Augment the log dir to add a subdir that is the env and seed
log_dir=${OUTPUTDIR}/${selected_env}_${seed}


echo " Running ${selected_env} with seed ${seed}"

echo "==Running sunrise.py=="

poetry run python OpenAIGym_SAC/examples/sunrise.py --seed=${seed} --log_dir=${log_dir} --env=${selected_env} --epochs=1000
echo "==sunrise.py Complete=="