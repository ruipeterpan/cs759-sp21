import os
import subprocess

CUDA_MPS_PIPE_DIRECTORY = '/tmp/nvidia-mps'
CUDA_MPS_LOG_DIRECTORY = '/tmp/nvidia-log'

os.environ['CUDA_MPS_PIPE_DIRECTORY'] = CUDA_MPS_PIPE_DIRECTORY
os.environ['CUDA_MPS_LOG_DIRECTORY'] = CUDA_MPS_LOG_DIRECTORY


def run_cmd(command, print_output=False):
    # os.system(command)
    output = subprocess.run(command, 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=True,
                            shell=True)
    if print_output:
        print(output.stdout.decode('utf-8').strip())



def start_server():
    """The nvidia-cuda-mps-server instances are created on-demand 
    when client applications connect to the control daemon.
    This function uses cudaMalloc to start the MPS server.
    """
    try: # check if a control daemon is spawned
        get_pid("nvidia-cuda-mps-control")
    except subprocess.CalledProcessError: # control daemon is not yet started
        start_daemon() # start the daemon
    dirname = os.path.dirname(os.path.realpath(__file__))
    cudaMalloc_path = os.path.join(dirname, "cudaMalloc")
    cudaMalloc_source_path = os.path.join(dirname, "cudaMalloc.cu")
    assert(os.path.exists(cudaMalloc_source_path)) # program that invokes cudaMalloc
    if not os.path.exists(cudaMalloc_path): # compile the program on the fly
        run_cmd("nvcc cudaMalloc.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o cudaMalloc")
    # print(f"Running command {cudaMalloc_path}")
    run_cmd(cudaMalloc_path) # run the command to start the MPS server


def start_daemon():
    """Start the MPS control daemon in background process, 
    assuming the user has enough privilege (e.g. root).
    """
    run_cmd("nvidia-cuda-mps-control -d")


def quit_everything():
    """Quit the MPS control daemon and any existing MPS servers.
    """
    run_cmd("echo quit | nvidia-cuda-mps-control")


def get_mps_server_pid():
    """Get the pid of the MPS server process.
    If the MPS server is not running, start one.
    Returns:
        int: pid of MPS server process
    """
    try:
        return get_pid("nvidia-cuda-mps-server")
    except subprocess.CalledProcessError: # server is not up
        start_server()
        return get_pid("nvidia-cuda-mps-server")


def get_pid(name):
    """Get the pid of a process by name.
    Args:
        name (str): Name of the process.
    Returns:
        int: pid of process
    """
    output = subprocess.check_output(["pidof", "-s", name])
    return int(output)


def set_active_thread_percentage_of_server(pid, percentage):
    """[summary]
    Args:
        pid (int): process id of the MPS server
        percentage (int): active thread percentage to set to
    """
    run_cmd(
        f"echo set_active_thread_percentage {pid} {percentage} | nvidia-cuda-mps-control")

def set_active_thread_percentage(percentage):
    """[summary]
    Args:
        percentage (int): active thread percentage to set to
    """
    server_pid = get_pid("nvidia-cuda-mps-server")
    set_active_thread_percentage_of_server(server_pid, percentage)


def get_active_thread_percentage_of_server(pid):
    output = subprocess.check_output(f"echo get_active_thread_percentage {pid} | nvidia-cuda-mps-control", shell=True)
    output = output.decode('utf-8').rstrip()
    return int(float(output))

def get_active_thread_percentage():
    server_pid = get_pid("nvidia-cuda-mps-server")
    return get_active_thread_percentage_of_server(server_pid)
