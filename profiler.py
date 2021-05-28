import argparse
import os
import psutil
import statistics
import subprocess
import traceback
import pandas as pd

from pynvml.smi import nvidia_smi
from time import sleep

LOG_DIR = "./tmp/"

def main(args):
    """Runs a workload using nvprof to collect core utilizations
    and produces a core-specific utilization score for the workload.

    Args:
        args (Namespace): Arguments
    """
    print("="*50)
    print(f"[Experiment {args.expr_name}] Profiling core-specific utilizations")
    print(f"[Experiment {args.expr_name}] Command for executing the workload: {args.command}")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(f"{LOG_DIR}{args.expr_name}", exist_ok=True)

    # run the workload without using nvprof to collect the NVML memory/utilization metrics
    print(f"[Experiment {args.expr_name}] Collecting NVML utilization")
    run_nvml(args)

    # run nvprof using the summary mode and metric mode and store the output file in /tmp/expr_name
    print(f"[Experiment {args.expr_name}] Running nvprof summary mode")
    run_nvprof(args, mode="summary")
    print(f"[Experiment {args.expr_name}] Running nvprof metric mode")
    run_nvprof(args, mode="metric")

    # parse the output files
    print(f"[Experiment {args.expr_name}] Parsing output files")
    summary_df = parse_file(args, mode="summary")
    metric_df = parse_file(args, mode="metric")
    merged_df = merge_df(summary_df, metric_df, args.expr_name)

    # calculates the core-specific utilization of a workload
    fp16_util, fp32_util, fp64_util = calculate_score(merged_df, args.expr_name)

    print("[Experiment {}] Utilization scores (FP16, FP32, FP64): ({:.2f}%, {:.2f}%, {:.2f}%)".format(
        expr_name, fp16_util, fp32_util, fp64_util))



def run_nvml(args):
    """Run a workload and use NVML to query its
    memory/compute usage

    Args:
        args (Namespace): Arguments
    """
    proc = subprocess.Popen(args.command.split(" "), 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    pid = proc.pid
    proc = psutil.Process(pid)

    nvsmi = nvidia_smi.getInstance()
    sleep(8)  # wait for training job to get started (set up model, dataloader, etc.)

    gpu_utils = []
    mems = []

    while True:
        if proc.status() == psutil.STATUS_ZOMBIE:
            # zombie/defunct prorcess
            break
        u = nvsmi.DeviceQuery()['gpu'][0]
        mem = u['fb_memory_usage']['used']
        util = u['utilization']
        gpu_util = util['gpu_util']
        gpu_utils.append(gpu_util)
        mems.append(mem)

    print(f"[Experiment {args.expr_name}] Avg GPU Util: {round(statistics.mean(gpu_utils), 2)}%")
    print(f"[Experiment {args.expr_name}] Avg mem: {round(statistics.mean(mems), 2)}MB")


def run_nvprof(args, mode):
    """Run the workload using nvprof

    Args:
        args (Namespace): Arguments
        mode (str): One of the two nvprof modes, either "summary" or "metric"
    """

    """
    Note that when using the nvprof metric mode, there may be security mechanisms
    that prevent non-sudo users from collecting the low-level metrics.
    There are two workarounds:
    (1) Have a sudoer modify the permissions
    (2) Run nvprof with sudo
    When using (2), "python3" in the training command may need to be replaced with the 
    absolute path to the python you are using. Also, running with sudo may result in some
    environment variables not getting picked up correctly.
    """
    if mode == "summary":
        command = f"nvprof --csv --log-file {LOG_DIR}{args.expr_name}/summary.csv -f {args.command}" 
    elif mode == "metric":
        command = f"nvprof -m tensor_precision_fu_utilization,single_precision_fu_utilization,double_precision_fu_utilization " \
            f"--csv --log-file {LOG_DIR}{args.expr_name}/metric.csv -f {args.command}"
        # command = f"sudo nvprof -m tensor_precision_fu_utilization,single_precision_fu_utilization,double_precision_fu_utilization " \
            # f"--csv --log-file {LOG_DIR}{args.expr_name}/metric.csv -f {args.command}"
    try:
        output = subprocess.run(command, 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                check=True,
                                shell=True)
        output = output.stdout.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        traceback.print_exc()
        exit(1)


def parse_file(args, mode):
    """Read from an nvprof output csv file,
    parses output to a dataframe

    Args:
        args (Namespace): Arguments
        mode (str): One of the two nvprof modes, either "summary" or "metric"
    """
    filename = f"{LOG_DIR}{args.expr_name}/{mode}.csv"
    if not os.path.exists(filename):
        print(f"[Experiment {args.expr_name}] {mode} file not found at {filename}, exiting")
        exit(1)

    # remove initial rows with profiling information
    f = open(filename, 'r')
    Lines = f.readlines()
    Lines = [x for x in Lines if "==" not in x]  # starts with ==pid==
    f.close()
    f = open(filename, 'w')
    f.writelines(Lines)
    f.close()

    if mode == "summary":
        df = pd.read_csv(filename)
        df = df[df['Type'] == "GPU activities"]  # drop API activities
        df = df.drop(["Type", "Min", "Max"], axis=1)  # drop useless columns
        result = df.rename(columns={"Time(%)": "time_%", "Time": "time_ms",
                        "Calls": "calls", "Avg": "avg_ms", "Name": "name"})
    elif mode == "metric":
        df = pd.read_csv(filename)
        df = df.drop(["Device", "Metric Description",
                        "Min", "Max", "Invocations"], axis=1)  # drop useless columns
        df = df.rename(columns={"Avg": "avg_util",
                        "Kernel": "name", "Metric Name": "metric_name"})
        # parse the utilization produced by nvprof
        # Example: Idle (0) -> 0, Mid (5)  -> 5
        df['avg_util'] = df['avg_util'].map(lambda x: int(x[-2:-1]))

        # add columns for tensor_util, single_util, double_util
        column_names = ["name", "tensor_util", "single_util", "double_util"]
        # forgive me for my sins of looping through a dataframe
        result = pd.DataFrame(columns=column_names)
        utils = [0, 0, 0]  # tensor, single, double
        for index, row in df.iterrows():
            utils[index % 3] = row["avg_util"]
            if index % 3 == 2:
                # append to result dataframe
                new_df = pd.DataFrame(
                    [[row["name"]] + utils], columns=column_names)
                result = result.append(new_df)
    return result


def merge_df(summary_df, metric_df, expr_name=None):
    """Merge two output dataframe from nvprof summary mode and
    nvprof metric mode into one dataframe

    Args:
        summary_df (DataFrame): Dataframe containing output from nvprof summary mode
        metric_df (DataFrame): Dataframe containing output from nvprof metric mode
        expr_name (str, optional): Unique name of workload. Defaults to None.

    Returns:
        DataFrame: Merged dataframe
    """
    df = pd.merge(metric_df, summary_df, how="outer", on="name")
    # replace mem operation's utilization (NaN because they are
    # in the summary but not the metric report) with 0
    df = df.fillna(0)
    df = df.sort_values(by=["time_%"], ascending=False)
    convert_type = {
        'time_%': 'float64',
        'time_ms': 'float64',
        'calls': 'int32',
        'avg_ms': 'float64',
    }
    df = df.astype(convert_type)
    if expr_name is not None:
        df.to_csv(f"{LOG_DIR}{expr_name}/merged.csv")
    return df


def calculate_score(df, expr_name):
    """For a merged df, calculate the utilization
    score of FP16, FP32, and FP64 for a workload

    Args:
        df (DataFrame): Merged dataframe of the outputs from summary mode and metric mode
        expr_name (str): Unique experiment name/ID of a workload
    """
    tensor_util_score = (df['tensor_util'] * df['time_ms']
                         ).sum() / df['time_ms'].sum()
    single_util_score = (df['single_util'] * df['time_ms']
                         ).sum() / df['time_ms'].sum()
    double_util_score = (df['double_util'] * df['time_ms']
                         ).sum() / df['time_ms'].sum()
    tensor_util_limit = (df['tensor_util'] * df['time_ms']
                         ).sum() / (10 * df['time_ms']).sum()
    single_util_limit = (df['single_util'] * df['time_ms']
                         ).sum() / (10 * df['time_ms']).sum()
    double_util_limit = (df['double_util'] * df['time_ms']
                         ).sum() / (10 * df['time_ms']).sum()
    return (100 * tensor_util_limit, 100 * single_util_limit, 100 * double_util_limit)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profiler parser wrapped around nvprof that reports the core-specific utilizations of a workload"
    )
    parser.add_argument('-id', '--expr-name', required=True, type=str,
                        help="Unique experiment name/ID of a workload")
    # parser.add_argument('-c', '--command', required=True, type=str, nargs='+',
    parser.add_argument('-c', '--command', required=True, type=str,
                        help="Command to execute the workload")
    args = parser.parse_args()
    # args.command = ' '.join(args.command)
    main(args)

if __name__ == "__main__":
    parse_args()