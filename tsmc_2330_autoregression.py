"""
AIoT-DA L6 Problem 2: TSMC (2330.TW) Stock Price Prediction using Auto Regression
Following CRISP-DM Methodology Workflow
"""

import os
import sys
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 避免 Windows PowerShell cp950 編碼問題
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("CRISP-DM 專案實作 (題目二)：台積電 (2330.TW) 股票價格預測 (Auto Regression 自迴歸模型)")
print("=" * 80)

# =====================================================================
# Phase 1: Business Understanding (商業理解)
# =====================================================================
print("\n[Phase 1: Business Understanding (商業理解)]")
print("目標：利用過去歷史價格（自迴歸 Lag 特徵與均線）預測台積電 (2330.TW) 次日收盤價。")
print("應用場景：量化投資交易策略、持股風險控管。")
print("成功標準：高方向準確度與低 RMSE 價格誤差。")

# =====================================================================
# Phase 2: Data Understanding (資料理解)
# =====================================================================
print("\n[Phase 2: Data Understanding (資料理解)]")
data_path = "2330_TW.csv"
if not os.path.exists(data_path):
    import yfinance as yf
    print("下載台積電 (2330.TW) 過去一年歷史股價資料...")
    df_raw = yf.download("2330.TW", period="1y", interval="1d")
    df_raw.to_csv(data_path)
else:
    print(f"讀取本地歷史股價檔案: {data_path}")

# 處理 yfinance 下載的多重索引標頭
df = pd.read_csv(data_path, header=[0, 1], index_col=0)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 確保只取前 5 個標準價格欄位
cols_needed = ['Open', 'High', 'Low', 'Close', 'Volume']
df = df[[c for c in cols_needed if c in df.columns]].dropna()
df.index = pd.to_datetime(df.index)
print(f"資料筆數: {len(df)} 交易日 (時間範圍: {df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')})")
print("\n--- 近期 5 個交易日收盤行情 ---")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail())

# =====================================================================
# Phase 3: Data Preparation (資料準備：特徵工程與時序切分)
# =====================================================================
print("\n[Phase 3: Data Preparation (資料準備：時間序列特徵工程)]")

# 1. 建立自迴歸滯後特徵 (Lagged Close Features)
df_feat = pd.DataFrame(index=df.index)
df_feat['Close_t'] = df['Close']
df_feat['Close_lag1'] = df['Close'].shift(1)
df_feat['Close_lag2'] = df['Close'].shift(2)
df_feat['Close_lag3'] = df['Close'].shift(3)
df_feat['Close_lag5'] = df['Close'].shift(5)

# 2. 建立技術指標特徵 (移動平均線 MA)
df_feat['MA5'] = df['Close'].rolling(window=5).mean()
df_feat['MA20'] = df['Close'].rolling(window=20).mean()
df_feat['Daily_Return'] = df['Close'].pct_change()

# 3. 移除因平移產生的空值
df_clean = df_feat.dropna()
feature_cols = ['Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag5', 'MA5', 'MA20']

X = df_clean[feature_cols]
y = df_clean['Close_t']

# 4. 時序切分 (前 80% 作為訓練，後 20% 作為測試，時間序列絕不可打亂洗牌)
split_idx = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"特徵清單: {feature_cols}")
print(f"訓練期交易日: {len(X_train)} 天 | 測試期交易日: {len(X_test)} 天")

# =====================================================================
# Phase 4: Modeling (模型建立：自迴歸 Auto Regression)
# =====================================================================
print("\n[Phase 4: Modeling (模型建立)]")
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

print(f"截距項 (b0): {model.intercept_:.2f}")
print("自迴歸權重係數 (Coefficients):")
for col, coef in zip(feature_cols, model.coef_):
    print(f"  - {col:12s}: {coef:8.4f}")

# =====================================================================
# Phase 5: Evaluation (模型評估與時序視覺化)
# =====================================================================
print("\n[Phase 5: Evaluation (模型評估)]")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"訓練集 R2 : {train_r2:.4f}")
print(f"測試集 R2 : {test_r2:.4f}")
print(f"測試集 RMSE: ${test_rmse:.2f} NTD")
print(f"測試集 MAE : ${test_mae:.2f} NTD")

# 繪製真實走勢 vs 自迴歸預測走勢圖
plt.figure(figsize=(12, 6))
test_dates = y_test.index
plt.plot(test_dates, y_test.values, label='實際收盤價 (Actual Close)', color='#2563EB', lw=2)
plt.plot(test_dates, y_pred_test, label=f'自迴歸預測價 (AR Predicted Close, R2={test_r2:.2f})', color='#EF4444', lw=2, linestyle='--')
plt.title('台積電 (2330.TW) 股價自迴歸預測對比圖 (Actual vs Auto-Regressive Predicted)', fontsize=14, fontweight='bold')
plt.xlabel('日期 (Date)', fontsize=12)
plt.ylabel('股價 (NTD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

stock_plot_path = "tsmc_2330_stock_prediction.png"
plt.savefig(stock_plot_path, dpi=300)
plt.close()
print(f"\n[股價預測對比圖表已生成]: {stock_plot_path}")

# =====================================================================
# Phase 6: Deployment (次日股價即時預測推論)
# =====================================================================
print("\n[Phase 6: Deployment (次日股價即時推論)]")
latest_features = pd.DataFrame([{
    'Close_lag1': df_clean['Close_t'].iloc[-1],
    'Close_lag2': df_clean['Close_t'].iloc[-2],
    'Close_lag3': df_clean['Close_t'].iloc[-3],
    'Close_lag5': df_clean['Close_t'].iloc[-5],
    'MA5': df_clean['Close_t'].iloc[-5:].mean(),
    'MA20': df_clean['Close_t'].iloc[-20:].mean()
}], columns=feature_cols)

next_day_pred = model.predict(latest_features)[0]
latest_close = df_clean['Close_t'].iloc[-1]
diff = next_day_pred - latest_close

print(f"最新收盤價: ${latest_close:.2f} NTD")
print(f"次日預測收盤價: ${next_day_pred:.2f} NTD (預期變動: {diff:+.2f} NTD, {diff/latest_close*100:+.2f}%)")

# 封裝模型
with gzip.open("tsmc_2330_ar_model.pkl.gz", "wb") as f:
    pickle.dump({'model': model, 'features': feature_cols}, f)
print("模型封裝完成: tsmc_2330_ar_model.pkl.gz")
print("=" * 80)
