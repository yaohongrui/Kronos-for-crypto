import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from model import Kronos, KronosTokenizer, KronosPredictor

def fetch_and_save_ohlcv(symbol_choice, timeframe):
    """
    从 OKX 获取指定交易对和时间段的 OHLCV 数据，
    将其保存为 CSV 文件，并返回一个 DataFrame。
    """
    exchange = ccxt.okx()
    
    # 构建永续合约的交易对符号
    symbol = f"{symbol_choice}-USDT-SWAP"
    
    print(f"Fetching data for {symbol} with timeframe {timeframe}...")

    limit_per_call = 300
    total_needed = 400
    all_ohlcv = []
    since = None

    while len(all_ohlcv) < total_needed:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_call)
        except ccxt.BaseError as e:
            print(f"Error fetching data: {e}")
            print(f"Please check if '{symbol}' and '{timeframe}' are valid on OKX.")
            return None

        if not ohlcv:
            print(f"Failed to fetch more data. Might have reached the beginning of {symbol} history.")
            break
            
        all_ohlcv = ohlcv + all_ohlcv
        since = ohlcv[0][0]

    final_ohlcv = all_ohlcv[-total_needed:]
    
    if not final_ohlcv:
        print("No data was retrieved. Please check your inputs.")
        return None

    df = pd.DataFrame(final_ohlcv, columns=['timestamps', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms')
    
    # 修改文件名生成逻辑以符合要求
    file_path = f"E:\\Kronos\\data\\{symbol_choice.lower()}.csv"
    df.to_csv(file_path, index=False)
    print(f"Successfully fetched and saved {len(final_ohlcv)} candlestick data points to {file_path}")
    
    return df

def run_prediction_and_plot(df, file_name):
    """
    运行 Kronos 预测模型并绘制结果。
    """
    # 1. 加载分词器和模型
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

    # 2. 初始化预测器
    predictor = KronosPredictor(model, tokenizer, device="cuda", max_context=512)

    # 3. 准备输入数据
    # 确保时间戳是正确的 pandas datetime 类型
    df['timestamps'] = pd.to_datetime(df['timestamps']) 

    lookback = 400
    pred_len = 25
    num_samples = 50  # 增加循环次数，以获得更稳定的统计结果

    # 提取输入数据
    x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume']]
    x_timestamp = df.loc[:lookback-1, 'timestamps']

    # 自动生成预测时间戳
    last_timestamp = x_timestamp.iloc[-1]

    time_interval = pd.Timedelta(minutes=15) 
    y_timestamp = pd.Series([last_timestamp + time_interval * i for i in range(1, pred_len + 1)])

    # 4. 生成多次预测并存储结果
    all_predictions = []
    print(f"开始生成 {num_samples} 次预测...")

    for i in range(num_samples):
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )
        all_predictions.append(pred_df['close'].values)

    # 5. 对预测结果进行统计
    all_predictions_array = np.array(all_predictions)
    # 计算每个时间点的中位数作为预测线
    median_prices = np.median(all_predictions_array, axis=0)
    # 计算每个时间点的10%和90%分位数，即80%置信区间
    q_10 = np.percentile(all_predictions_array, 10, axis=0)
    q_90 = np.percentile(all_predictions_array, 90, axis=0)

    # 6. 可视化
    plt.figure(figsize=(12, 6))

    # 绘制真实数据
    plt.plot(df['timestamps'][:lookback], df['close'][:lookback], label='Previous Price', color='blue')

    # 绘制预测中位数
    plt.plot(y_timestamp, median_prices, label='Median Forecast', color='orange', linestyle='--')

    # 绘制80%置信区间
    plt.fill_between(y_timestamp, q_10, q_90, color='orange', alpha=0.3, label='80% Confidence Interval')

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Price Forecast and 80% Confidence Interval')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    """
    主函数，协调数据抓取和预测任务。
    """
    # 设置默认值
    default_symbol = 'ETH'
    default_timeframe = '15m'

    # 获取用户输入，并设置默认值
    symbol_input = input(f"Enter symbol (default: {default_symbol}): ").upper()
    symbol_choice = symbol_input if symbol_input else default_symbol
    
    timeframe_input = input(f"Enter timeframe (default: {default_timeframe}): ").lower()
    timeframe = timeframe_input if timeframe_input else default_timeframe
    
    # 1. 抓取数据
    df = fetch_and_save_ohlcv(symbol_choice, timeframe)
    
    if df is not None and not df.empty:
        # 2. 运行预测和绘图
        run_prediction_and_plot(df, symbol_choice.lower())
    else:
        print("Data fetching failed. Exiting.")

if __name__ == "__main__":
    main()