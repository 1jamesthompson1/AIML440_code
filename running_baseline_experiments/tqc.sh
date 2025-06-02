#!/bin/bash

cd ~/code/tqc_pytorch
task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id

# A problem with the initial grid running attmept meant about 2/3 failed. These are the task ids that failed.
needed_tasks=(6 7 8 10 11 12 14 16 17 18 19 20 21 22 23 24 25 26 27 28 29 31 33 34 35 40 41 42 43 44 45 46 47 48 49 50 51 53 54 57 58 59 60 62 64 65 66 67 68 69 75 76 77 78 79 80 81 82 83 84 85 88 89)
# Convert the task id to the task id that is needed.
task_id=${needed_tasks[$task_id]}



num_seeds=10
num_algos=3
num_envs=9

runs_per_algo=$(($num_seeds * $num_envs))

seed=$(($task_id % $num_seeds))
env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

envs=("HalfCheetah-v5" "Walker2d-v5" "Humanoid-v5" "Ant-v5" "HumanoidStandup-v5" "Swimmer-v5" "Hopper-v5" "InvertedDoublePendulum-v5" "Pusher-v5")

selected_env=${envs[$env_index]}

echo " Running ${selected_env} with seed ${seed}"
echo "==Running main.py=="
poetry run python main.py --seed=${seed} --log_dir=$OUTPUTDIR --env=${selected_env} --save_model --max_timesteps=3000000
echo "==main.py Complete=="
