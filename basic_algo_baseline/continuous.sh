# Run the discrete_train file from the workbench directory

cd ~/code/AIML440_code/basic_algo_baseline

echo "==Working Directory=="
pwd
echo "==Listing Files=="
ls

echo "==Understanding which algorithm to run=="
task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id
num_seeds=10
num_algos=3
num_envs=9

runs_per_algo=$(($num_seeds * $num_envs))

seed=$(($task_id % $num_seeds))
algo_index=$(( ($task_id / $runs_per_algo) % $num_algos))
env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

algorithms=("ppo" "sac" "td3")
selected_algo=${algorithms[$algo_index]}

echo " Running ${selected_algo} in env ${env_index} with seed ${seed}"

echo "==Running Script=="

poetry run python train.py --seed=${seed} --output_dir=$OUTPUTDIR --algo=${selected_algo} --env_index=${env_index}

echo "==Script Complete=="