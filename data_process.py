import pandas as pd

# 1. 读取你的原始数据
df = pd.read_csv('E:\Kronos\data\BTC_usdt_swap_15m.csv')

# 2. 确保数据是按时间排序的
df.sort_values(by='timestamps', inplace=True)

# 3. 计算未来收益率作为预测目标
# 假设我们想预测未来15分钟的收益率
# 我们使用 shift(-1) 来获取下一条K线的收盘价，然后计算收益
df['future_return'] = df['close'].shift(-1) / df['close'] - 1

# 4. 删除最后一行（因为它的 future_return 是 NaN）
# 并删除任何其他可能存在的空值
df.dropna(inplace=True)

# 5. 保存处理好的数据为新的CSV文件
# 这个文件将用于下一步的 Qlib 数据预处理
df.to_csv('eth_processed_data.csv', index=False)

print("数据处理完成，已保存为 eth_processed_data.csv")