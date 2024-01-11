import matplotlib.pyplot as plt

# 資料
X = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
Y = [73.46, 73.22, 72.49, 71.70, 70.72, 69.40, 68.43, 66.35, 64.71, 63.59, 61.40, 60.10, 58.34]

# 畫圖
plt.plot(X, Y, marker="o", linestyle="-", color="b")

# 加上標題和軸標籤
# plt.title('Relative Training Noise vs. Mean IoU')
plt.xlabel("Relative Noise n_tr = n_inf (%)")
plt.ylabel("Mean IoU (%)")

# 儲存圖片
plt.savefig("fig_C.png")
