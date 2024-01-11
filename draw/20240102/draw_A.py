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

# 對相同的X值做平均
combined_data = {}
for x, y1, y2 in zip(data1["X"], data1["Y"], data2["Y"]):
    if x not in combined_data:
        combined_data[x] = []
    combined_data[x].append((y1 + y2) / 2)

# 將結果拆分回兩個list以便繪製
average_data = {"X": [], "Y": []}
for x, y_list in combined_data.items():
    average_data["X"].append(x)
    average_data["Y"].append(np.mean(y_list))

# 繪製藍色和綠色的散點
plt.scatter(data1["X"], data1["Y"], marker="o", label="First Experiment", color="blue")
plt.scatter(data2["X"], data2["Y"], marker="o", label="Second Experiment", color="forestgreen")

# 繪製黃色的虛線
plt.plot(average_data["X"], average_data["Y"], marker="", linestyle="-", label="Average", color="darkorange")

# 加上標題和軸標籤
# plt.title('Relative Training Noise vs. Mean IoU')
plt.xlabel("Relative Training Noise n_tr (%)")
plt.ylabel("Mean IoU (%)")

# 顯示圖例
plt.legend()

# 儲存圖片
plt.savefig("fig_A.png")
