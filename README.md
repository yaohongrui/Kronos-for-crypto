# Kronos 量化预测项目

本项目基于深度学习模型 Kronos，实现了加密货币（如 BTC/ETH/SOL）K线数据的自动抓取、归一化处理、行情预测与可视化。支持本地模型和分词器加载，适合量化研究与策略开发。

## 效果演示
BTCUSDT15m
![example](https://github.com/yaohongrui/Kronos-for-crypto/blob/main/example/example.png)

## 目录结构

```
.
├── main.py                             # 主入口
├── data/                               # 存放抓取的K线数据
├── local_models/                       # 本地模型与分词器
├── model/                              # 模型核心代码
├── outputs/                            # 预测结果与图表
```

## 快速开始

1. **安装依赖**
   - 安装 Python 3.10+，然后安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```
   主要依赖：`torch`, `ccxt`, `pandas`, `numpy`, `matplotlib`, `tqdm`, `einops`, `huggingface_hub`

2. **运行主程序**
   ```bash
   python main.py
   ```

3. **查看结果**
   - 预测结果和图表会保存在 `outputs/` 目录。

## 主要功能

- 自动抓取 Binance 期货 K线数据，自动去除不完整K线
- 支持多币种、多时间周期
- 基于 Kronos 模型的行情预测
- TP/SL 风险管理与可视化
- 预测结果多样采样与统计分析

## 配置说明

- 配置参数集中在各主程序的 `Config` 类中，可根据需求调整。
- 本地模型路径、数据保存路径等请确保存在且有读写权限。


## 关于使用

- 关键位置方向指示准确度较高
- 不要乱开仓，小仓位大饼1000个点左右可以加仓，其他的类推
- 使用方面自己不会的问gemini
- 不要扛单，不要扛单，不要扛单
- 模型结果仅供参考
- TP/SL看看就好，自己可以改

