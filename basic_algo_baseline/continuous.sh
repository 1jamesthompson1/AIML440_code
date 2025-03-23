# Run the discrete_train file from the workbench directory

cd ~/code/AIML440_code/basic_algo_baseline

echo "==Working Directory=="
pwd
echo "==Listing Files=="
ls

echo "==Running Script=="

poetry run python continuous_train.py

echo "==Script Complete=="