import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from model import Kronos, KronosTokenizer, KronosPredictor
from tqdm import tqdm
import os

def fetch_and_save_ohlcv(symbol_choice, timeframe):
    """
    从 Binance 获取指定交易对和时间段的 OHLCV 数据，
    将其保存为 CSV 文件，并返回一个 DataFrame。
    """
    # 使用 Binance 交易所
    exchange = ccxt.binance()
    
    # 构建永续合约的交易对符号（Binance 格式）
    # Binance 的永续合约符号是像 'BTCUSDT' 这样的，没有 -SWAP 后缀
    symbol = f"{symbol_choice.upper()}USDT"
    
    print(f"Fetching data for {symbol} with timeframe {timeframe}...")

    limit_per_call = 1000  # Binance 的限制通常是 1000
    total_needed = 350
    all_ohlcv = []
    since = None

    while len(all_ohlcv) < total_needed:
        try:
            # Binance 的 fetch_ohlcv 调用
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_call)
        except ccxt.BaseError as e:
            print(f"Error fetching data: {e}")
            print(f"Please check if '{symbol}' and '{timeframe}' are valid on Binance.")
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
    
    file_path = f"./data/{symbol_choice.lower()}.csv"
    
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df.to_csv(file_path, index=False)
    print(f"Successfully fetched and saved {len(final_ohlcv)} candlestick data points to {file_path}")
    
    return df

def run_prediction_and_plot(df, file_name, timeframe):
    """
    运行 Kronos 预测模型并绘制结果。
    """
    # 1. 加载分词器和模型
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

    # 2. 初始化预测器
    predictor = KronosPredictor(model, tokenizer, device="cuda", max_context=512)

    # 3. 准备输入数据
    df['timestamps'] = pd.to_datetime(df['timestamps']) 

    lookback = 350
    pred_len = 15
    num_samples = 50

    x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume']]
    x_timestamp = df.loc[:lookback-1, 'timestamps']

    last_timestamp = x_timestamp.iloc[-1]

    # 根据时间周期获取时间间隔，这里假设时间戳是按固定间隔排列的
    if len(x_timestamp) > 1:
        time_interval = x_timestamp.iloc[1] - x_timestamp.iloc[0]
    else:
        # 如果只有一个数据点，则默认一个15分钟的间隔
        time_interval = pd.Timedelta(minutes=15)
        
    y_timestamp = pd.Series([last_timestamp + time_interval * i for i in range(1, pred_len + 1)])

    # 4. 生成多次预测并存储结果
    all_predictions = []
    print(f"开始生成 {num_samples} 次预测...")

    # 使用 tqdm 包装 range(num_samples) 来显示进度条
    for i in tqdm(range(num_samples), desc="Generating predictions"):
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
    median_prices = np.median(all_predictions_array, axis=0)
    q_10 = np.percentile(all_predictions_array, 10, axis=0)
    q_90 = np.percentile(all_predictions_array, 90, axis=0)

    # 6. 可视化
    plt.figure(figsize=(12, 6))
    history_plot_len = 150
    
    # 确保历史数据切片不会超出可用范围
    start_index = max(0, lookback - history_plot_len)
    
    # 绘制历史价格曲线
    plt.plot(df['timestamps'][start_index:lookback], df['close'][start_index:lookback], label='Previous Price', color='blue')
    
    # 绘制预测价格曲线和置信区间
    plt.plot(y_timestamp, median_prices, label='Median Forecast', color='orange', linestyle='--')
    plt.fill_between(y_timestamp, q_10, q_90, color='orange', alpha=0.3, label='80% Confidence Interval')

    # 使用 AutoDateLocator 和 AutoDateFormatter 自动优化刻度
    locator = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # 自动调整 x 轴刻度标签，防止重叠
    plt.gcf().autofmt_xdate()

    plt.xlabel('Time')
    plt.ylabel('Price')
    # 使用 symbol_choice 和 timeframe 更新标题
    plt.title(f'Price Forecast for {file_name.upper()}-USDT ({timeframe})')
    plt.legend()
    plt.grid(True)
    
    # --- 新增的保存代码 ---
    # 生成带有当前时间戳、交易对名称和时间周期的唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 这里我们使用 file_name (symbol_choice) 和 timeframe
    save_path = f"./outputs/{file_name.upper()}_{timeframe}.png"
    
    # 确保 plots 目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        plt.savefig(save_path)
        print(f"成功将预测图保存至: {save_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    # 显示图像
    plt.show()

def main():
    """
    主函数，协调数据抓取和预测任务。
    """
    default_symbol = 'BTC'
    default_timeframe = '15m'

    symbol_input = input(f"Enter symbol (default: {default_symbol}): ").upper()
    symbol_choice = symbol_input if symbol_input else default_symbol
    
    timeframe_input = input(f"Enter timeframe (default: {default_timeframe}): ").lower()
    timeframe = timeframe_input if timeframe_input else default_timeframe
    
    df = fetch_and_save_ohlcv(symbol_choice, timeframe)
    
    if df is not None and not df.empty:
        # 在这里，我们将 timeframe 变量传递给 run_prediction_and_plot
        run_prediction_and_plot(df, symbol_choice.lower(), timeframe)
    else:
        print("Data fetching failed. Exiting.")

if __name__ == "__main__":
    main()