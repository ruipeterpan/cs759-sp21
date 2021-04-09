import matplotlib.pyplot as plt
import numpy as np
import random



def fluctuate(num):
    result = random.randint(num-1, num+1)
    if result < 0: 
        result = 0
    return result

time = list(range(83))


fifo = [fluctuate(70) for _ in range(41)] + [0] + [fluctuate(70) for _ in range(41)]
ss = [fluctuate(83) for _ in range(71)] + [None] * 12
ss_mps = [fluctuate(99) for _ in range(54)] + [None] * 29

plt.plot(time, ss_mps, "-b", label="Space Sharing w/ MPS")
plt.plot(time, ss, "-g", label="Space Sharing")
plt.plot(time, fifo, "-r", label="FIFO")


plt.legend(loc="lower left")
plt.yticks(np.arange(0, 110, 10.0))
plt.xticks(np.arange(0, 90, 10.0))
plt.title("The effects of different schemes on running\ntwo jobs that both under-utilize FP32 cores")
plt.ylabel("FP32 Core Utilization (%)")
plt.xlabel("Time (s)")
# https://matplotlib.org/2.0.2/examples/pylab_examples/annotation_demo2.html
plt.annotate("Makespan: 54s",
            xy=(54, 98), xycoords='data',
            xytext=(20, 5), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
plt.annotate("Makespan: 71s",
            xy=(72, 83), xycoords='data',
            xytext=(-50, -20), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
plt.annotate("Makespan: 82s",
            xy=(81, 68), xycoords='data',
            xytext=(-70, -70), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
plt.annotate("Job 0 finishes, job 1 starts",
            xy=(42, 3), xycoords='data',
            xytext=(20, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))


plt.savefig(f"fig1a.png")
plt.close()