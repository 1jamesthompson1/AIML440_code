#!bin/bash

# This is going to be used to run the CrossQ and DroQ algorithms on a individual computer.

cd ~/code/AIML440_code/running_baseline_experiments
echo "==Working Directory=="
pwd
echo "==Listing Files=="
ls

unique_id=$(date +%Y%m%d_%H%M%S)
output_folder=~/grid-output/${unique_id}_sbx
mkdir -p $output_folder/logs

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${num_gpus} GPUs."

# Limit the number of concurrent jobs
max_jobs=${num_gpus}

function run_experiment() {
    task_id=$(($1-1)) # 0-indexed except that grid system doesnt allow 0 as the task id
    output_dir=$2
    gpu_id=$3
    num_seeds=10
    num_algos=2
    num_envs=9

    echo "==Understanding which algorithm to run=="
    runs_per_algo=$(($num_seeds * $num_envs))

    seed=$(($task_id % $num_seeds))
    algo_index=$(( ($task_id / $runs_per_algo) % $num_algos))
    env_index=$(( (($task_id % $runs_per_algo) / $num_seeds) % $num_seeds))

    algorithms=("droq" "crossq")
    selected_algo=${algorithms[$algo_index]}

    echo " Running ${selected_algo} in env ${env_index} with seed ${seed}"

    start_time=$(date +%s)
    echo "======================================"
    echo "  Running training script: $1"
    echo "  Start time: $(date)"
    echo "  Start seconds: $(start_time)"
    echo "======================================"
    export JAX_PLATFORMS=cpu
    CUDA_VISIBLE_DEVICES='' poetry run python train.py --seed=${seed} --output_dir=${output_dir} --algo=${selected_algo} --env_index=${env_index} --time_steps=12000 --verbosity=1

    echo "======================================"
    echo "  Finished training script"
    echo "  End time: $(date)"
    echo "  Elapsed seconds: $(( $(date +%s) - start_time ))"
    echo "====================================="

    sleep 10
}

declare -a gpu_jobs
for ((i=0; i<num_gpus; i++)); do
    gpu_jobs[$i]=0
done

function wait_for_free_gpu() {
    while true; do
        for ((i=0; i<num_gpus; i++)); do
            if [[ ${gpu_jobs[$i]} -eq 0 ]]; then
                echo $i
                return
            fi
        done
        sleep 5
    done
}


for i in {90..180..28}
do
    gpu_id=0
    echo "==Running task ${i} on GPU: ${gpu_id}=="

    gpu_jobs[$gpu_id]=1  # mark as busy

    (
        run_experiment ${i} ${output_folder} ${gpu_id} \
            > ${output_folder}/logs/${i}_stdout.txt \
            2> ${output_folder}/logs/${i}_stderr.txt
        gpu_jobs[$gpu_id]=0  # mark as free when done
    ) &

done

echo "All scripts are running in the background. Check the logs in ${output_folder}/logs/ for output."
