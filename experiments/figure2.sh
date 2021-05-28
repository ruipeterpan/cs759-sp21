# Figure 2

# start NVIDIA MPS daemon thread
nvidia-cuda-mps-control -d

mps_server_pid=55117  # this should be the pid of the MPS server process
percentage=50  # set MPS thread percentage here

echo set_active_thread_percentage $mps_server_pid $percentage | nvidia-cuda-mps-control

# run the workload and record the output JCT. Divide 50000 (size of CIFAR10) using the JCT to get the throughput
python3 ./workloads/cv_benchmark.py --model resnet18 --dataset cifar10 --batch-size 16 --epochs 1
