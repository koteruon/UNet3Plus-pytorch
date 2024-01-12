import matplotlib.pyplot as plt
import numpy as np

# 第一組資料
data1 = {
    "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Y": [73.46, 73.55, 73.68, 73.77, 73.81, 74.06, 74.04, 73.36, 73.32, 73.11, 72.90, 72.11, 72.13],
}

# 第二組資料
data2 = {
    "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Y": [72.61, 73.19, 73.07, 73.12, 72.65, 73.25, 73.00, 72.14, 72.38, 72.24, 72.05, 72.42, 71.90],
}

# 第二組資料
data3 = {
    "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Y": [73.19, 73.20, 73.41, 73.67, 73.32, 73.56, 73.36, 73.53, 73.67, 72.90, 72.60, 72.45, 72.03],
}

# Calculate mean and standard error of the mean (SEM) for each X value
mean_y_values = []
sem_y_values = []

for x in data1["X"]:
    y_values = (
        [data1["Y"][i] for i, val in enumerate(data1["X"]) if val == x]
        + [data2["Y"][i] for i, val in enumerate(data2["X"]) if val == x]
        + [data3["Y"][i] for i, val in enumerate(data3["X"]) if val == x]
    )

    mean_y = np.mean(y_values)
    sem_y = np.std(y_values) / np.sqrt(len(y_values))

    mean_y_values.append(mean_y)
    sem_y_values.append(sem_y)

# Plot horizontal lines representing the mean with error bars for each X value
plt.errorbar(data1["X"], mean_y_values, yerr=sem_y_values, fmt="s", capsize=5, markersize=6, linestyle="-")

# Add a horizontal dashed line for average at X-axis 0
plt.axhline(y=mean_y_values[0], color="gray", linestyle="--")

# Set axis labels
plt.xlabel("Relative Training Noise n_tr (%)")
plt.ylabel("Mean IoU (%)")

# 儲存圖片
plt.savefig("fig_A.png")
