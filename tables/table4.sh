# Table 4

# Prerequisites: CUDA, Python3.8, CUDA-compatible PyTorch, CUDA-compatible torchvision.
# To run jobs concurrently using NVIDIA MPS, an NVIDIA GPU that is compatible with MPS is required.
# The results in the reported are measured using the Volta MPS architecture.
# Not using a Volta GPU may result in the degradation of results
# There are also ~10% variations in the numbers across runs


##########################################
# Row 1
##########################################


python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r1_j1.txt


##########################################
# Row 2: 
##########################################

# start NVIDIA MPS daemon thread
nvidia-cuda-mps-control -d

# TODO: run some workload to start MPS server

# set this variable to be the pid of the current MPS server
mps_server_pid=55117

echo set_active_thread_percentage $mps_server_pid 50 | nvidia-cuda-mps-control

python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r2_j1.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r2_j2.txt 2>&1 &


##########################################
# Row 2: 
##########################################

# start NVIDIA MPS daemon thread
nvidia-cuda-mps-control -d

# TODO: run some workload to start MPS server

# set this variable to be the pid of the current MPS server
mps_server_pid=55117

echo set_active_thread_percentage $mps_server_pid 33 | nvidia-cuda-mps-control

python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r3_j1.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r3_j2.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r3_j3.txt 2>&1 &


##########################################
# Row 2: 
##########################################

# start NVIDIA MPS daemon thread
nvidia-cuda-mps-control -d

# TODO: run some workload to start MPS server

# set this variable to be the pid of the current MPS server
mps_server_pid=55117

echo set_active_thread_percentage $mps_server_pid 25 | nvidia-cuda-mps-control

python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r4_j1.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r4_j2.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r4_j3.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r4_j4.txt 2>&1 &



##########################################
# Row 2: 
##########################################

# start NVIDIA MPS daemon thread
nvidia-cuda-mps-control -d

# TODO: run some workload to start MPS server

# set this variable to be the pid of the current MPS server
mps_server_pid=55117

echo set_active_thread_percentage $mps_server_pid 14 | nvidia-cuda-mps-control

python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r7_j1.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r7_j2.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r7_j3.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r7_j4.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r7_j5.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r7_j6.txt 2>&1 &
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1 > ./output/t4_r7_j7.txt 2>&1 &






# stop the MPS server and daemon
echo quit | nvidia-cuda-mps-control