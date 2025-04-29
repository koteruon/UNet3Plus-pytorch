import matplotlib.pyplot as plt
import numpy as np

# 输入数据
X1 = 11.0
Y1 = [72.11, 72.78, 72.14, 72.16, 72.42]

X2 = 12.0
Y2 = [72.13, 72.16, 72.13, 72.28, 71.90]

# 计算Y的平均值
mean_Y1 = np.mean(Y1)
mean_Y2 = np.mean(Y2)

# 绘制数据点
plt.scatter([X1] * len(Y1), Y1, color="blue", label="X = 11.0")
plt.scatter([X2] * len(Y2), Y2, color="green", label="X = 12.0")

# 绘制平均值线
plt.axhline(y=mean_Y1, xmin=-0.1, xmax=0.1, color="blue", linestyle="--", label=f"Mean Y (X = 11.0): {mean_Y1:.2f}")
plt.axhline(y=mean_Y2, xmin=0.9, xmax=1.1, color="green", linestyle="--", label=f"Mean Y (X = 12.0): {mean_Y2:.2f}")

# 添加文本标注
plt.text(X1 + 0.4, mean_Y1, f"Mean IoU (X = 11.0): {mean_Y1:.2f}", color="blue", ha="right", va="bottom")
plt.text(X2 - 0.05, mean_Y2, f"Mean IoU (X = 12.0): {mean_Y2:.2f}", color="green", ha="right", va="bottom")

# 加上標題和軸標籤
# plt.title('Relative Training Noise vs. Mean IoU')
plt.xlabel("Relative Training Noise n_tr (%)")
plt.ylabel("Mean IoU (%)")

# 儲存圖片
plt.savefig("fig_A.png")
