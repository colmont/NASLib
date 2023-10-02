optimizer=bananas
predictors=(gpwl) # gpwl

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=naslib
current_datetime=$(date +%Y%m%d_%H%M%S)
out_dir="playground/runs/bo/${current_datetime}"
mkdir -p $out_dir
cp naslib/runners/predictors/gp_config.yaml $out_dir

# search space / data:
search_space=nasbench101
dataset=cifar10
search_epochs=30

# trials / seeds:
trials=1
end_seed=$(($start_seed + $trials - 1))

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    python scripts/create_configs.py --predictor $predictor \
    --epochs $search_epochs --start_seed $start_seed --trials $trials \
    --out_dir $out_dir --dataset=$dataset --config_type nas_predictor \
    --search_space $search_space --optimizer $optimizer
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for predictor in ${predictors[@]}
    do
        config_file=$out_dir/$dataset/configs/nas_predictors/config\_$optimizer\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python naslib/runners/nas_predictors/runner.py --config-file $config_file
    done
done
