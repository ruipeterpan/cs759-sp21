# CS/ECE/ME/EP 759 Spring 2021 Final Project Code Base

This README contains the instructions to replicate the results in Rui Pan's final project report:
Cautiously Aggressive GPU Space Sharing in Large-Scale Multi-Tenant Clusters.

Some of the prerequisites for replicating the results include:

- An NVIDIA GPU with Volta architecture
- Python 3.8 nightly build
- CUDA-compatible PyTorch & TorchVision

## pymps: Provides Python access to NVIDIA CUDA Multi-Process Service (MPS)

The code is open-sourced at `https://github.com/ruipeterpan/pymps`.

A local codebase is included as a submodule at `./pymps`.

## Table 1: Utilization of common deep learning / high performance computing workloads

For the workloads in row 1 (resnet18) and row 4 (cuBLAS), the profiling outputs of nvprof can be found at `./scripts/output/{cublas_dgemm, resnet18}_{metric, summary, merged}.csv`. For the workloads in row 2 (resnet18 quantization), the profiling outputs are the files ending with `.csv` at `./scripts/output/`

The merged metric file can be produced using `./scripts/parse_profiling_results.py`. Instructions for producing the profiling outputs are also included in this Python file. A not organized guide on the nvprof commands to run can be found at `./table1.sh`.

## Figure 1 & Table 2: Space-sharing workloads that use different types of cores

See `./table2.sh` for detailed instructions. The scripts and source graph for figure 1 is included in `./images`.

## Table 3: The statistics of different sharing schemes.

See `./table3.sh` for detailed instructions.

## Figure 2: Throughput of training ResNet-18 on CIFAR10 using different MPS thread percentages

See `./figure2.sh` for detailed instructions.

## Table 4. The benefits of packing >2 jobs concurrently.

See `./table4.sh` for detailed instructions.
