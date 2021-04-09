import matplotlib.pyplot as plt
import numpy as np
import random



def fluctuate(num):
    result = random.randint(num-1, num+1)
    if result < 0: 
        result = 0
    return result

time = list(range(90))


fp32_util = [fluctuate(0) for _ in range(44)] + [fluctuate(25), fluctuate(50)] + [fluctuate(75) for _ in range(44)]
fp64_util = [fluctuate(85) for _ in range(44)] + [fluctuate(56), fluctuate(28)] + [fluctuate(0) for _ in range(44)]


plt.plot(time, fp32_util, "-r", label="FP32 Util")
plt.plot(time, fp64_util, "-b", label="FP64 Util")
plt.legend(loc="upper right")
plt.yticks(np.arange(0, 110, 10.0))
plt.xticks(np.arange(0, 100, 10.0))
plt.title("Running two jobs that utilizes different\ntypes of cores sequentially")
plt.ylabel("Core Utilization (%)")
plt.xlabel("Time (s)")
# https://matplotlib.org/2.0.2/examples/pylab_examples/annotation_demo2.html
plt.annotate("Makespan: 90s",
            xy=(90.0, 75), xycoords='data',
            xytext=(-100, -100), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
plt.annotate("Job 0 starts",
            xy=(0, 83), xycoords='data',
            xytext=(50, -100), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
plt.annotate("Job 1 starts",
            xy=(47, 69), xycoords='data',
            xytext=(20, -50), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))

plt.savefig(f"fig1_fifo.png")
plt.close()


fp32_util = [fluctuate(75) for _ in range(53)] + [0] * 37
fp64_util = [fluctuate(85) for _ in range(53)] + [0] * 37


plt.plot(time, fp32_util, "-r", label="FP32 Util")
plt.plot(time, fp64_util, "-b", label="FP64 Util")
plt.legend(loc="upper right")
plt.yticks(np.arange(0, 110, 10.0))
plt.xticks(np.arange(0, 100, 10.0))
plt.title("Packing two jobs that utilizes\ndifferent types of cores")
plt.ylabel("Core Utilization (%)")
plt.xlabel("Time (s)")
plt.annotate("Makespan: 53s",
            xy=(53.0, 80), xycoords='data',
            xytext=(50, -100), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
plt.annotate("Job 0 and 1 start at the same time",
            xy=(0, 68), xycoords='data',
            xytext=(10, -50), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
plt.savefig(f"fig1_ss.png")
plt.close()