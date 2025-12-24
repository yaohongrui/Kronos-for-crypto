'''
# 更新日志

## 版本 2.2 (2025年12月15日)
- 优化了 TP/SL 计算逻辑，基于预测终点的标准差，提升风险管理准确性。
- 增加了交易方向自动判断功能，根据胜率概率决定做多或做空。
- 改进了图表绘制，TP/SL 线现在延伸至预测时间段末尾，提升可读性。
- 增加了当前价格水平线，作为参考点。
'''
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
# 假设这些模型类和分词器可以正常导入
from model import Kronos, KronosTokenizer, KronosPredictor
from tqdm import tqdm
import os
import time

# --- 配置区 ---
class Config:
    """集中管理所有硬编码参数"""
    # 路径配置 (请确保路径正确)
    LOCAL_MODEL_PATH = "./local_models/Kronos_base"
    LOCAL_TOKENIZER_PATH = "./local_models/Kronos_Tokenizer_base"
    DATA_DIR = "./data"
    OUTPUT_DIR = "./outputs"
    
    # 数据抓取配置
    EXCHANGE_OPTIONS = {'defaultType': 'future'} # Binance U本位永续合约
    # 额外增加一根K线，用于在 fetch_and_save_ohlcv 中舍弃掉不完整的K线
    TOTAL_CANDLES_NEEDED = 350 + 1 
    LIMIT_PER_CALL = 1000       # 每次API调用最大限制
    MAX_RETRIES = 3             # 网络错误最大重试次数
    RETRY_DELAY = 5             # 网络错误重试间隔 (秒)

    # 预测配置
    LOOKBACK = 350 
    PRED_LEN = 15
    NUM_SAMPLES = 64 # 建议平时用64，做重要决策时可开到100+
    HISTORY_PLOT_LEN = 100 # 为了看清TP/SL线，稍微缩短一点历史绘图长度
    DEVICE = 'cuda'
    MAX_CONTEXT = 512
    
    # 策略配置 (基于标准差的倍数)
    SL_MULTIPLIER = 2.0  # 止损 = 0.75倍标准差
    TP_MULTIPLIER = 2.0  # 止盈 = 1倍标准差
# --- 配置区结束 ---

# 全局变量
global KRONOS_MODEL
global KRONOS_TOKENIZER
KRONOS_MODEL = None
KRONOS_TOKENIZER = None


def load_kronos_components():
    """纯本地加载分词器和模型"""
    global KRONOS_MODEL, KRONOS_TOKENIZER
    if KRONOS_MODEL is not None and KRONOS_TOKENIZER is not None:
        return True
    
    try:
        KRONOS_TOKENIZER = KronosTokenizer.from_pretrained(Config.LOCAL_TOKENIZER_PATH)
        KRONOS_MODEL = Kronos.from_pretrained(Config.LOCAL_MODEL_PATH)
        return True
    except Exception as e:
        print("❌ 模型或分词器从本地加载失败！请检查路径和文件。")
        KRONOS_MODEL = None
        KRONOS_TOKENIZER = None
        return False


def fetch_and_save_ohlcv(symbol_choice, timeframe):
    """从 Binance 获取 OHLCV 数据"""
    exchange = ccxt.binance({'options': Config.EXCHANGE_OPTIONS})
    symbol = f"{symbol_choice.upper()}/USDT"
    
    all_ohlcv = []
    since = None
    retries = Config.MAX_RETRIES
    target_count = Config.TOTAL_CANDLES_NEEDED 
    
    with tqdm(total=target_count, desc=f"Fetching {symbol} ({timeframe}) OHLCV", leave=False) as pbar:
        while len(all_ohlcv) < target_count:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=since, 
                    limit=Config.LIMIT_PER_CALL
                )
            except ccxt.NetworkError as e:
                if retries > 0:
                    retries -= 1
                    time.sleep(Config.RETRY_DELAY)
                    continue
                else:
                    print(f"\n❌ [Error] 达到最大重试次数。")
                    return None
            except ccxt.BaseError as e:
                print(f"\n❌ [Error] CCXT 错误: {e}")
                return None
            
            if not ohlcv:
                break
                
            all_ohlcv = ohlcv + all_ohlcv
            since = ohlcv[0][0]
            pbar.update(len(ohlcv))
            retries = Config.MAX_RETRIES

    final_ohlcv_temp = all_ohlcv[-target_count:]
    
    if len(final_ohlcv_temp) > 0:
        final_ohlcv = final_ohlcv_temp[:Config.LOOKBACK]
    else:
        final_ohlcv = []

    if len(final_ohlcv) < Config.LOOKBACK:
        print(f"\n❌ 未获取到 {Config.LOOKBACK} 根完整的 K 线数据。")
        return None

    df = pd.DataFrame(final_ohlcv, columns=['timestamps', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamps'] = pd.to_datetime(df['timestamps'], unit='ms')
    
    file_path = os.path.join(Config.DATA_DIR, f"{symbol_choice.lower()}_{timeframe}.csv")
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    df.to_csv(file_path, index=False)
    
    return df


def run_prediction_and_plot(df, symbol_name, timeframe):
    """运行预测，计算统计数据，并绘制包含 TP/SL 的图表。"""
    
    if KRONOS_MODEL is None or KRONOS_TOKENIZER is None:
        return

    predictor = KronosPredictor(
        KRONOS_MODEL, 
        KRONOS_TOKENIZER, 
        device=Config.DEVICE, 
        max_context=Config.MAX_CONTEXT
    )

    # 1. 数据准备
    df['timestamps'] = pd.to_datetime(df['timestamps']) 
    
    lookback = Config.LOOKBACK
    pred_len = Config.PRED_LEN
    num_samples = Config.NUM_SAMPLES

    x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    x_timestamp = df.iloc[-lookback:]['timestamps'].reset_index(drop=True)

    last_timestamp = x_timestamp.iloc[-1]
    if len(x_timestamp) > 1:
        time_interval = x_timestamp.diff().mode().iloc[0] 
    else:
        time_interval = pd.Timedelta(minutes=15) 
        
    y_timestamp = pd.Series([last_timestamp + time_interval * i for i in range(1, pred_len + 1)])

    # 2. 预测循环
    all_predictions = []
    
    # 保持 T=0.8, top_p=0.9 以获得健康的波动率
    T_VALUE = 0.8 
    TOP_P_VALUE = 0.6

    for i in tqdm(range(num_samples), desc="Generating predictions", leave=False):
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T_VALUE,        
            top_p=TOP_P_VALUE,
            sample_count=1, 
            verbose=False
        )
        all_predictions.append(pred_df['close'].values.flatten())

    # 3. 统计与量化指标
    all_predictions_array = np.array(all_predictions)
    
    current_close = df['close'].iloc[-1]
    final_preds = all_predictions_array[:, -1]
    
    # 计算方向概率
    total_samples = len(final_preds)
    bullish_count = np.sum(final_preds > current_close)
    prob_up = (bullish_count / total_samples) * 100
    prob_down = 100 - prob_up

    # --- 核心：计算波动率和 TP/SL ---
    # 使用所有样本“最终点”的标准差，这代表了预测周期结束时的风险分布
    sigma = np.std(final_preds)
    
    # 自动判断交易方向
    trade_direction = "LONG" if prob_up >= 50 else "SHORT"
    
    if trade_direction == "LONG":
        sl_price = current_close - (sigma * Config.SL_MULTIPLIER)
        tp_price = current_close + (sigma * Config.TP_MULTIPLIER)
        # 如果标准差太小导致止盈比现价还低（极端情况），做个兜底
        if tp_price <= current_close: tp_price = current_close * 1.005
    else: # SHORT
        sl_price = current_close + (sigma * Config.SL_MULTIPLIER)
        tp_price = current_close - (sigma * Config.TP_MULTIPLIER)
        if tp_price >= current_close: tp_price = current_close * 0.995

    # 计算期望值相关
    bullish_samples = final_preds[final_preds > current_close]
    bearish_samples = final_preds[final_preds <= current_close]
    avg_gain = np.mean(bullish_samples) - current_close if len(bullish_samples) > 0 else 0
    avg_loss = current_close - np.mean(bearish_samples) if len(bearish_samples) > 0 else 0
    ev = (prob_up / 100 * avg_gain) - (prob_down / 100 * avg_loss)

    # 4. 打印报告
    print(f"\n" + "="*45)
    print(f"📊 交易计划 (基于波动率 Sigma={sigma:.2f})")
    print(f"="*45)
    print(f"当前价格: {current_close:.2f}")
    print(f"建议方向: {'🟢 做多 (LONG)' if trade_direction == 'LONG' else '🔴 做空 (SHORT)'}")
    print(f"胜率概率: {prob_up if trade_direction == 'LONG' else prob_down:.1f}%")
    print(f"-"*45)
    print(f"🎯 止盈 (TP): {tp_price:.2f} (+{Config.TP_MULTIPLIER}σ)")
    print(f"🛡️ 止损 (SL): {sl_price:.2f} (-{Config.SL_MULTIPLIER}σ)")
    print(f"-"*45)
    print(f"💰 期望值 (EV): {ev:+.2f} USDT")
    print(f"="*45 + "\n")

    # 5. 可视化绘图
    median_prices = np.median(all_predictions_array, axis=0)
    q_10 = np.percentile(all_predictions_array, 10, axis=0)
    q_90 = np.percentile(all_predictions_array, 90, axis=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    
    # 绘制历史价格 (截取一段以便看清)
    start_index = max(0, len(df) - Config.HISTORY_PLOT_LEN)
    history_df_plot = df.iloc[start_index:]
    ax.plot(history_df_plot['timestamps'], history_df_plot['close'], label='History', color='blue', linewidth=1.5)
    
    # 绘制预测曲线
    ax.plot(y_timestamp, median_prices, label='Median Forecast', color='orange', linestyle='--', linewidth=2)
    ax.fill_between(y_timestamp, q_10, q_90, color='orange', alpha=0.25, label='80% Confidence Interval')

    # --- 绘制 TP 和 SL 线 ---
    # 为了美观，线画在预测时间段内
    line_start = y_timestamp.iloc[0]
    line_end = y_timestamp.iloc[-1] + time_interval * 2 # 稍微延长一点

    # 止盈线 (绿色)
    ax.hlines(y=tp_price, xmin=line_start, xmax=line_end, colors='green', linestyles='dashdot', linewidth=2, label=f'TP: {tp_price:.0f}')
    # 止损线 (红色)
    ax.hlines(y=sl_price, xmin=line_start, xmax=line_end, colors='red', linestyles='dotted', linewidth=2, label=f'SL: {sl_price:.0f}')
    
    # 在图表右侧添加价格标签
    ax.text(line_end, tp_price, f' TP\n {tp_price:.0f}', color='green', verticalalignment='center', fontweight='bold')
    ax.text(line_end, sl_price, f' SL\n {sl_price:.0f}', color='red', verticalalignment='center', fontweight='bold')
    
    # 当前价格水平线 (灰色参考)
    ax.axhline(y=current_close, color='gray', linestyle='-', alpha=0.3, linewidth=1)

    # 格式化坐标轴
    locator = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    plt.xlabel('Time')
    plt.ylabel('Price')
    direction_icon = "🚀" if trade_direction == "LONG" else "🩸"
    plt.title(f'{direction_icon} Plan: {symbol_name.upper()}-{timeframe} | Dir: {trade_direction} | Win: {max(prob_up, prob_down):.1f}%')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.5)
    
    save_file_name = f"{symbol_name.upper()}_{timeframe}_Plan.png"
    save_path = os.path.join(Config.OUTPUT_DIR, save_file_name)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    try:
        plt.savefig(save_path)
        print(f"✅ 交易计划图已保存: {save_path}")
    except Exception as e:
        print(f"❌ 保存图像失败: {e}")

    plt.show()

def main():
    if not load_kronos_components(): return

    default_symbol = 'BTC'
    default_timeframe = '15m'

    symbol_input = input(f"Enter symbol (default: {default_symbol}): ").upper()
    symbol_choice = symbol_input if symbol_input else default_symbol
    
    timeframe_input = input(f"Enter timeframe (default: {default_timeframe}): ").lower()
    timeframe = timeframe_input if timeframe_input else default_timeframe
    
    df = fetch_and_save_ohlcv(symbol_choice, timeframe)
    
    if df is not None and not df.empty:
        run_prediction_and_plot(df, symbol_choice.lower(), timeframe)
    else:
        print("\n❌ 错误：无法获取数据。")

if __name__ == "__main__":
    main()