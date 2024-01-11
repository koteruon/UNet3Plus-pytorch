import matplotlib.pyplot as plt

# 資料
X = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
Y = [73.46, 73.55, 73.68, 73.77, 73.81, 74.06, 74.04, 73.36, 73.32, 73.11, 72.90, 72.32, 72.12]

# 畫圖
plt.plot(X, Y, marker="o", linestyle="-", color="b")

# 加上標題和軸標籤
# plt.title('Relative Training Noise vs. Mean IoU')
plt.xlabel("Relative Training Noise n_tr (%)")
plt.ylabel("Mean IoU (%)")

# 儲存圖片
plt.savefig("fig_A.png")
