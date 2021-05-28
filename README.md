# CS/ECE/ME/EP 759 Spring 2021 Final Project

This README contains the code base for Rui Pan's final project report: Cautiously Aggressive GPU Space Sharing in Multi-Tenant Clusters.

Some of the prerequisites for replicating the results include:

* An NVIDIA GPU with Volta architecture
* Python 3.8 nightly build
* CUDA-compatible PyTorch & TorchVision

This repo contains:

* `/data`: Source data for running the workloads. It should be set up as follows:
	* `/imagenet`: [ImageNet Dataset](https://image-net.org/download-images.php) for resnet50 workloads
	* `/ml-20m`: [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/) for recommendation/recoder workloads
	* `wikitext2`: [WikiText-2 Dataset](https://github.com/pytorch/examples/tree/master/word_language_model/data/wikitext-2) for language modeling workloads
* `/latex`: LaTex files for editing the report on Overleaf
* `/output`: Core-specific utilizations of workloads produced using an earlier version of the profiler
* `/tables`: Shell scripts for replicating the profiling results in various tables
* `/workloads`: Common DL/HPC workloads used in the evaluations. A lot of these are copied from [Gavel](https://github.com/stanford-futuredata/gavel).
* `plotting.ipynb`: Jupyter Notebook that produces all figures in the report
* `profiler.py`: Profiler parser wrapped around nvprof
* `pymps.py`: Provides Python access to NVIDIA CUDA Multi-Process Service (MPS)
* `README.md`: [Well, of course I know him. He's me.](kenobi.jpg)
* `report.pdf`: PDF version of the final report