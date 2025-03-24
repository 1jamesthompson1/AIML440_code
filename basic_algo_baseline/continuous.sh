# Run the discrete_train file from the workbench directory

cd ~/code/AIML440_code/basic_algo_baseline

echo "==Working Directory=="
pwd
echo "==Listing Files=="
ls

echo "==Understanding which algorithm to run=="

seed=$((${1} % 10))
algo_index=$(( (${1} / 10) % 10))

algorithms=("ppo" "sac" "td3")
selected_algo=${algorithms[$algo_index]}

echo " Running ${selected_algo} with seed ${seed}"

echo "==Running Script=="

poetry run python train.py --seed=${seed} --output_dir=$OUTPUTDIR --algo=${selected_algo}

echo "==Script Complete=="