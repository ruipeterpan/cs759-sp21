# Table 2: Space-sharing workloads that use different types of cores.

# Prerequisites: CUDA, Python3.8, CUDA-compatible PyTorch, CUDA-compatible torchvision.
# To run jobs concurrently using NVIDIA MPS, an NVIDIA GPU that is compatible with MPS is required.
# The results in the reported are measured using the Volta MPS architecture.
# Not using a Volta GPU may result in the degradation of results
# There are also ~10% variations in the numbers across runs

# compile the cublas dgemm workload to simulate hpc workloads
nvcc cublas_dgemm.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o cublas_dgemm

##########################################
# Row 1: FIFO
# Run the two workloads sequentially
# For ease of verification, we reduce the time of the two workloads by 10x
##########################################

# Job 0: 45.89s
./workloads/cublas_dgemm 512 500000 > ./output/t2_r1_j1.txt

# Job 1: 43.24s
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t2_r1_j2.txt


##########################################
# Row 2: Space sharing w/o MPS
# Run the two workloads concurrently
# For ease of verification, we reduce the time of the two workloads by 10x
##########################################

# packing w/o mps: 88s, 49.35s (hpc has a significant slowdown after dl is started)
./workloads/cublas_dgemm 512 500000 > ./output/t2_r2_j1.txt 2>&1 & python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t2_r2_j2.txt 2>&1 &

# use the Linux watch utility to check the command line output of nvidia-smi
watch -d -n 0.1 nvidia-smi


##########################################
# Row 3: Space sharing with MPS
# Run the two workloads concurrently
# For ease of verification, we reduce the time of the two workloads by 10x
##########################################

# start NVIDIA MPS daemon thread
nvidia-cuda-mps-control -d

# packing w/ mps: 59s, 58s
./workloads/cublas_dgemm 512 500000 > ./output/t2_r3_j1.txt 2>&1 & python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t2_r3_j2.txt 2>&1 &

# use the Linux watch utility to check the command line output of nvidia-smi
watch -d -n 0.1 nvidia-smi

# stop the MPS server and daemon
echo quit | nvidia-cuda-mps-control