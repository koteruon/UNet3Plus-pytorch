import matplotlib.pyplot as plt
import pandas as pd

# 假設你的資料存放在一個名為 'data.csv' 的 CSV 檔案中
df = pd.read_csv("weight_track.csv")

# 根據首行數值進行分組，選擇首行數值相同的資料
grouped = df.groupby(df.iloc[:, 0])

# 畫出每組資料的變化曲線
plt.figure(figsize=(10, 6))
for name, group in grouped:
    plt.plot(group.columns[1:], group.iloc[0, 1:], label=f"Group {name}")

plt.xlabel("Column Index")
plt.ylabel("Values")
plt.title("Variation Curves for Rows with Same First Value")
plt.legend()
plt.show()
