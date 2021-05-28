# Table 1: Per-core utilizations of common deep learning / high performance computing workloads

# Prerequisites: CUDA, Python3.8 nightly version (for CUDA AMP), CUDA-compatible PyTorch, CUDA-compatible torchvision, nvprof
# The results in the reported are measured on a Tesla V100.
# There are ~10% variations in the numbers across runs.

# compile the cublas dgemm workload to simulate hpc workloads
nvcc ./workloads/cublas_dgemm.cu ./workloads/mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o ./workloads/cublas_dgemm

##########################################
# Row ?: ???
##########################################

# (Summary file) ResNet-18 on CIFAR, quantization (mixed precision training), estimated run time: ~120s
nvprof --csv --log-file ./output/resnet18_quantized_summary.csv -f \
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 --cuda-amp

# (Metric file) ResNet-18 on CIFAR, quantization (mixed precision training), estimated run time: ???s
nvprof -m tensor_precision_fu_utilization,single_precision_fu_utilization,double_precision_fu_utilization \
--csv --log-file ./output/resnet18_quantized_metric.csv -f \
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 --cuda-amp --log-interval 10

##########################################
# Row ?: ???
##########################################

# (Summary file) ResNet-18 on CIFAR, quantization (mixed precision training), estimated run time: ~120s
nvprof --csv --log-file ./output/resnet18_summary.csv -f \
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1

# (Metric file) ResNet-18 on CIFAR, quantization (mixed precision training), estimated run time: ???s
nvprof -m tensor_precision_fu_utilization,single_precision_fu_utilization,double_precision_fu_utilization \
--csv --log-file ./output/resnet18_metric.csv -f \
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 --log-interval 10