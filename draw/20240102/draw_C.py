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

# 顯示圖例
plt.legend()

# 加上標題和軸標籤
# plt.title('Relative Training Noise vs. Mean IoU')
plt.xlabel("Relative Noise n_tr = n_inf (%)")
plt.ylabel("Mean IoU (%)")

# 儲存圖片
plt.savefig("fig_C.png")
