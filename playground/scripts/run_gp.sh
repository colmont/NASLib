predictors=(gpwl gp_heat)

experiment_types=(single single)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=NASLib

# search space / data:
search_space=nasbench101
dataset=cifar10

# other variables:
trials=100
end_seed=$(($start_seed + $trials - 1))
test_size=200
train_size_single_values=(250 500)

n_approx_values=(10 25 50 100 250 500)

for train_size_single in ${train_size_single_values[@]}
do
    # Get current date and time in format YYYYMMDD_HHMMSS
    current_datetime=$(date +%Y%m%d_%H%M%S)

    # Generate output directory name
    out_dir="/cluster/scratch/cdoumont/playground/runs/gp/${current_datetime}"

    # Create output directory if it does not exist
    mkdir -p $out_dir

    # # Modify gp_config.yaml with the current n_approx value
    # python playground/scripts/modify_yaml.py $n_approx

    # Copy gp_config.yaml to output directory
    cp naslib/runners/predictors/gp_config.yaml $out_dir

    # create config files
    for i in $(seq 0 $((${#predictors[@]}-1)) )
    do
        predictor=${predictors[$i]}
        experiment_type=${experiment_types[$i]}
        python scripts/create_configs.py --predictor $predictor --experiment_type $experiment_type \
        --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
        --dataset=$dataset --config_type predictor --search_space $search_space --train_size_single $train_size_single
    done

    # run experiments
    for t in $(seq $start_seed $end_seed)
    do
        for predictor in ${predictors[@]}
        do
            config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
            echo ================running $predictor trial: $t =====================
            python naslib/runners/predictors/runner.py --config-file $config_file
        done
    done
done
