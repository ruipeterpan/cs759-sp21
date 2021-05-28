# Table 3

# compile the cublas dgemm workload to simulate hpc workloads
nvcc cublas_dgemm.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o cublas_dgemm

##########################################
# Row 1: FIFO
# Run the two workloads sequentially
##########################################

python3 ./workloads/cv_benchmark.py --model resnet50 --dataset imagenet --batch-size 64 --epochs 1

python3 ./workloads/cv_benchmark.py --model squeezenet --dataset svhn --batch-size 32 --epochs 4

# repeat the procedures in table2.sh

# to set the MPS percentage, use 
mps_server_pid=55117

echo set_active_thread_percentage $mps_server_pid 33 | nvidia-cuda-mps-control

# for row 4, (1) set percentage to be 90, (2) launch job 1, (3) set percentage to be 10, (4) launch job 2