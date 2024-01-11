import matplotlib.pyplot as plt
import numpy as np

# 第一組資料
data1 = {
    "X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Y": [73.46, 73.55, 73.68, 73.77, 73.81, 74.06, 74.04, 73.36, 73.32, 73.11, 72.90, 72.11, 72.13],
}

# 第二組資料
data2 = {
    "X": [0, 4, 11, 12],
    "Y": [73.27, 72.83, 71.52, 71.61],
}

# 對相同的X值做平均
combined_data = {"X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "Y": []}
for x in combined_data["X"]:
    y1, y2 = 0, 0
    non_zero = 0
    if x in data1["X"]:
        y1 = data1["Y"][data1["X"].index(x)]
        non_zero += 1
    if x in data2["X"]:
        y2 = data2["Y"][data2["X"].index(x)]
        non_zero += 1
    combined_data["Y"].append((y1 + y2) / non_zero)

# 繪製藍色和綠色的散點
plt.scatter(data1["X"], data1["Y"], marker="o", label="First Experiment", color="blue")
plt.scatter(data2["X"], data2["Y"], marker="o", label="Second Experiment", color="forestgreen")

# 繪製黃色的虛線
plt.plot(combined_data["X"], combined_data["Y"], marker="", linestyle="-", label="Average", color="darkorange")

# 加上標題和軸標籤
# plt.title('Relative Training Noise vs. Mean IoU')
plt.xlabel("Relative Training Noise n_tr (%)")
plt.ylabel("Mean IoU (%)")

# 顯示圖例
plt.legend()

# 儲存圖片
plt.savefig("fig_A.png")