import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

df = pd.read_csv('results_2D_inhouse_vs_rest_01.csv')

num_points = df["num_points"]

t1 = df["final_2D_robust--with_static_filters)"]
t2 = df["triangle--incremental"]
t3 = df["triangle--dnc"]
t4 = df["qhull"]


plt.grid(True)
plt.loglog(num_points[3:], t1[3:], linestyle='--', marker='.', linewidth=0.75)
plt.loglog(num_points[3:], t2[3:], linestyle='--', marker='.', linewidth=0.75)
plt.loglog(num_points[3:], t3[3:], linestyle='--', marker='.', linewidth=0.75)
plt.loglog(num_points[3:], t4[3:], linestyle='--', marker='.', linewidth=0.75)

plt.legend([
    # r"final\_2D (non-robust)",
    # r"final\_2D\_robust (no static filters)",
    r"final\_2D\_robust (with static filters)",
    r"Triangle (incremental)",
    r"Triangle (divide-and-conquer)",
    r"Qhull"
])

plt.xlabel(r"Number of Points", size=10)
plt.ylabel(r"Running Time $(s)$", size=10)
plt.suptitle(r"Results", size=13)

plt.savefig("results_2D_inhouse_vs_rest_02.png", dpi=300, bbox_to_inches="tight")