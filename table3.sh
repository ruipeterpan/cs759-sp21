# packing workloads that use different types of cores

# compile the cublas dgemm workload to simulate hpc workloads
nvcc cublas_dgemm.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o cublas_dgemm

# 85%, 45.89s
./cublas_dgemm 512 500000

# 75%, 43.24s
python3 cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1

# packing w/o mps: 88s, 49.35s (hpc has a significant slowdown after dl is started)
# packing w/ mps: 59s, 58s