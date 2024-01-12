import matplotlib.pyplot as plt
import numpy as np

# 第一組資料
data1 = {
    "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Y": [73.46, 73.22, 72.49, 71.70, 70.72, 69.40, 68.43, 66.35, 64.71, 63.59, 61.40, 60.10, 58.34],
}

# 第二組資料
data2 = {
    "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Y": [72.61, 72.79, 71.83, 70.80, 70.24, 69.35, 67.83, 66.22, 65.51, 63.54, 62.57, 61.44, 59.95],
}

# 第二組資料
data3 = {
    "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Y": [73.19, 72.92, 72.58, 71.59, 70.34, 69.39, 68.36, 67.22, 65.92, 63.97, 62.17, 60.12, 58.37],
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
# 加上標題和軸標籤
# plt.title('Relative Training Noise vs. Mean IoU')
plt.xlabel("Relative Noise n_tr = n_inf (%)")
plt.ylabel("Mean IoU (%)")

# 儲存圖片
plt.savefig("fig_C.png")
