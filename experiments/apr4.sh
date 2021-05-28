date="apr4"

pid=2569

# resnet18 on cifar10
PercentageList=" 100 90 80 70 60 50 40 30 20 10 "
ModelList=" resnet18 "
DatasetList=" cifar10 "
BatchSizeList=" 1024 512 256 128 64 32 16 "
OptLevelList=" O0 O1 "

for opt_level in $OptLevelList; do
    for dataset in $DatasetList; do
        for model in $ModelList; do
            for batchsize in $BatchSizeList; do
                for percentage in $PercentageList; do
                    EXPR_NAME="$dataset-$model-$batchsize-$opt_level-$percentage"
                    echo "============================"
                    echo "Running experiment $EXPR_NAME"
                    echo set_active_thread_percentage $pid $percentage | nvidia-cuda-mps-control
                    python3 cv_benchmark.py --model $model --dataset $dataset --batch-size $batchsize --epoch 1 --apex-amp $opt_level --no-test > ./log/$date/$EXPR_NAME.txt
                done
            done
        done
    done
done

# resnet50 on imagenet
ModelList=" resnet50 "
DatasetList=" imagenet "
BatchSizeList=" 128 64 32 16 "
OptLevelList=" O0 O1 "

for opt_level in $OptLevelList; do
    for dataset in $DatasetList; do
        for model in $ModelList; do
            for batchsize in $BatchSizeList; do
                for percentage in $PercentageList; do
                    EXPR_NAME="$dataset-$model-$batchsize-$opt_level-$percentage"
                    echo "============================"
                    echo "Running experiment $EXPR_NAME"
                    echo set_active_thread_percentage $pid $percentage | nvidia-cuda-mps-control
                    python3 cv_benchmark.py --model $model --dataset $dataset --batch-size $batchsize --epoch 1 --apex-amp $opt_level --no-test > ./log/$date/$EXPR_NAME.txt
                done
            done
        done
    done
done


ModelList=" recoder "
DatasetList=" ml20m "
BatchSizeList=" 16384 8192 4096 2048 1024 512 "
OptLevelList=" O0 O1 "

for opt_level in $OptLevelList; do
    for dataset in $DatasetList; do
        for model in $ModelList; do
            for batchsize in $BatchSizeList; do
                for percentage in $PercentageList; do
                    EXPR_NAME="$dataset-$model-$batchsize-$opt_level-$percentage"
                    echo "============================"
                    echo "Running experiment $EXPR_NAME"
                    echo set_active_thread_percentage $pid $percentage | nvidia-cuda-mps-control
                    python3 ~/pf-rui-shiv/experiments/gavel_workloads/recommendation/train.py --data_dir /home/ruipan/data/ml-20m/pro_sg/ --batch_size $batchsize --num_epochs 1 --apex-amp $opt_level > ./log/$date/$EXPR_NAME.txt
                done
            done
        done
    done
done


ModelList=" LSTM "
DatasetList=" wikitext2 "
BatchSizeList=" 320 160 80 40 20 10 5 "
OptLevelList=" O0 O1 "

for opt_level in $OptLevelList; do
    for dataset in $DatasetList; do
        for model in $ModelList; do
            for batchsize in $BatchSizeList; do
                for percentage in $PercentageList; do
                    EXPR_NAME="$dataset-$model-$batchsize-$opt_level-$percentage"
                    echo "============================"
                    echo "Running experiment $EXPR_NAME"
                    echo set_active_thread_percentage $pid $percentage | nvidia-cuda-mps-control
                    python3 ~/pf-rui-shiv/experiments/gavel_workloads/language_modeling/main.py --cuda --model $model --data /home/ruipan/data/wikitext2 --batch_size $batchsize --epochs 1 --apex-amp $opt_level > ./log/$date/$EXPR_NAME.txt
                done
            done
        done
    done
done



ModelList=" Transformer "
DatasetList=" multi30k "
BatchSizeList=" 128 64 32 16 "
OptLevelList=" O0 O1 "

for opt_level in $OptLevelList; do
    for dataset in $DatasetList; do
        for model in $ModelList; do
            for batchsize in $BatchSizeList; do
                for percentage in $PercentageList; do
                    EXPR_NAME="$dataset-$model-$batchsize-$opt_level-$percentage"
                    echo "============================"
                    echo "Running experiment $EXPR_NAME"
                    echo set_active_thread_percentage $pid $percentage | nvidia-cuda-mps-control
                    python3 ~/pf-rui-shiv/experiments/gavel_workloads/translation/train.py -data /home/ruipan/data/translation/multi30k.atok.low.pt -batch_size $batchsize -epoch 1 -proj_share_weight --apex-amp $opt_level > ./log/$date/$EXPR_NAME.txt
                done
            done
        done
    done
done


ModelList=" CycleGAN "
DatasetList=" monet2photo "
BatchSizeList=" 4 2 1 "
OptLevelList=" O0 O1 "

for opt_level in $OptLevelList; do
    for dataset in $DatasetList; do
        for model in $ModelList; do
            for batchsize in $BatchSizeList; do
                for percentage in $PercentageList; do
                    EXPR_NAME="$dataset-$model-$batchsize-$opt_level-$percentage"
                    echo "============================"
                    echo "Running experiment $EXPR_NAME"
                    declare -i nsteps
                    nsteps=40/$batchsize
                    echo set_active_thread_percentage $pid $percentage | nvidia-cuda-mps-control
                    python3 ~/pf-rui-shiv/experiments/gavel_workloads/cyclegan/cyclegan.py --dataset_path /home/ruipan/data/monet2photo --decay_epoch 0 --n_steps $nsteps --batch_size $batchsize --apex-amp $opt_level > ./log/$date/$EXPR_NAME.txt
                done
            done
        done
    done
done


