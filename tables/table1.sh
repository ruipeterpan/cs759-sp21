# Table 1: Core-specific utilizations of common DL/HPC workloads

# Prerequisites: CUDA, Python3.8 nightly version (for CUDA AMP), CUDA-compatible PyTorch, CUDA-compatible torchvision, nvprof
# The results in the reported are measured on a Tesla V100.
# There are ~10% variations in the numbers across runs.

# If you are reading this, you'll notice that these measurements are done before developing the profiler wrapped around nvprof.
# Sorry about that!

# compile the cublas dgemm workload to simulate hpc workloads
# Using "real" HPC workloads like MACSio is on my todo list, although it's likely that I will be too lazy to come back to this
nvcc ../workloads/cublas_dgemm.cu ../workloads/mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o ../workloads/cublas_dgemm

# Template for producing a summary file
nvprof --csv --log-file FILENAME -f COMMAND

# Template for producing a metric file
nvprof -m tensor_precision_fu_utilization,single_precision_fu_utilization,double_precision_fu_utilization \
--csv --log-file FILENAME -f COMMAND


# Row 1: ResNet18, CIFAR-10
python3 ../workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size $batchsize --epochs 1
# Row 2: ResNet18, CIFAR-10, quantization with torch.cuda.amp.autocast()
python3 ../workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size $batchsize --epochs 1 --cuda-amp
# Row 3: ResNet50, ImageNet
python3 ../workloads/cv_benchmark.py --model resnet50  --dataset imagenet --batch-size $batchsize --epochs 1 --cuda-amp
# Row 4: LSTM, WikiText-2
python3 ../workloads/language_modeling/main.py --cuda --model LSTM --data /home/ruipan/data/wikitext2 --batch_size $batchsize --epochs 1
# Row 5: LSTM, WikiText-2, quantization
python3 ../workloads/language_modeling/main.py --cuda --model LSTM --data /home/ruipan/data/wikitext2 --batch_size $batchsize --epochs 1 --apex-amp $opt_level
# Row 6: Recommendation/Recoder, ML-20M
python3 ../workloads/recommendation/train.py --data_dir /home/ruipan/data/ml-20m/pro_sg/ --batch_size $batchsize --num_epochs 1 --apex-amp $opt_level
# Row 7: cuBLAS DGEMM
./cublas_dgemm 512 100