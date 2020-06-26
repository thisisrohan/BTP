import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

df = pd.read_csv('data.csv')
# print(df)

num_points = df["num_points"]

t1 = df["serial"]
t2 = df["parallel"]


plt.grid(True)
plt.loglog(num_points, t1, linestyle='--', marker='.', linewidth=0.75)
plt.loglog(num_points, t2, linestyle='--', marker='.', linewidth=0.75)

plt.legend([
    # r"final\_2D (non-robust)",
    # r"final\_2D\_robust (no static filters)",
    r"final\_2D (serial)",
    r"Parallel 2D triangulator"
])

plt.xlabel(r"Number of Points", size=10)
plt.ylabel(r"Running Time $(s)$", size=10)
plt.suptitle(r"Results", size=13)

plt.savefig("results_2D_inhouse_vs_rest_02.png", dpi=300, bbox_to_inches="tight")