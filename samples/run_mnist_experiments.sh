# long settings
n_test_interval=30
n_epoch=901
results="Paper_results_long/"

# short settings
# n_test_interval=1
# n_epoch=50
# results="Paper_results/"

cwd=`dirname "$0"`
program=${cwd}/pytorch_conv3L_mnist.py
seed=0
n_runs=1
n_test_samples=1000
n_iters_test=1000

tput setaf 2; echo "Results will be saved in ${results}"; tput setaf 9

while [ ${seed} -lt ${n_runs} ];
do
    tput setaf 2; echo "Running ${program} with seed ${seed}"; tput setaf 9
    ${program} --seed ${seed} --n_epochs ${n_epoch} --n_test_interval ${n_test_interval} --output ${results} --n_test_samples ${n_test_samples} --n_iters_test ${n_iters_test}
    seed=$((seed+1))
done
