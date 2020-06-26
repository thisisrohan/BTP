import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('results_3D_inhouse_vs_rest_01.csv')

num_points = df["num_points"]

t1 = df["final_3D"]
t2 = df["qhull"]

s1 = t2 / t1
# s2 = t3 / t1
# s3 = t4 / t1

plt.grid(True)
plt.semilogx(num_points, np.ones(shape=len(num_points)), color='k', linewidth=0.75)
plt.semilogx(num_points, s1, linestyle='--', marker='.', linewidth=0.75)
# plt.semilogx(num_points, s2, linestyle='--', marker='.', linewidth=0.75)
# plt.semilogx(num_points, s3, linestyle='--', marker='.', linewidth=0.75)

plt.legend([
    # r"final\_2D (non-robust)",
    # r"final\_2D\_robust (no static filters)",
    # r"final\_2D\_robust (with static filters)",
    r"Speed Up = 1.0",
#    r"Triangle (incremental)",
#    r"Triangle (divide-and-conquer)",
    r"Qhull"
])

plt.xlabel(r"Number of Points", size=10)
plt.ylabel(r"Speed Up", size=10)
plt.suptitle(r"Speed Up over different triangulators", size=13)

plt.savefig("speedup.png", dpi=300, bbox_to_inches="tight")