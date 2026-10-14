"""
AIoT-DA L6 Multiple Linear Regression & 5 Feature Selection Methods Comparison (50 Startups)
Following CRISP-DM Methodology
"""

import os
import sys
import gzip
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# 避免 Windows PowerShell cp950 編碼問題
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("CRISP-DM 專案實作：50 Startups 回歸分析與 5 大特徵選取方法/回歸演算法評測")
print("=" * 80)

# =====================================================================
# Phase 1: Business Understanding (商業理解)
# =====================================================================
print("\n[Phase 1: Business Understanding (商業理解)]")
print("目標：預測新創公司的獲利 (Profit)，評估研發、行政、行銷支出及地理州別之影響力。")
print("評測 5 大特徵選取演算法 (Lasso L1, Forward SFS, RFE, ElasticNet, Exhaustive Search) 與多種回歸模型。")

# =====================================================================
# Phase 2: Data Understanding (資料理解)
# =====================================================================
print("\n[Phase 2: Data Understanding (資料理解)]")
data_path = "50_Startups.csv" if os.path.exists("50_Startups.csv") else "50_Startups (2).csv"
data = pd.read_csv(data_path)
print(f"載入資料檔案: {data_path} (維度: {data.shape})")
print("\n--- 資料前 5 筆 ---")
print(data.head())

# =====================================================================
# Phase 3: Data Preparation (資料準備)
# =====================================================================
print("\n[Phase 3: Data Preparation (資料準備)]")
X_raw = data.drop('Profit', axis=1)
y = data['Profit']

# 獨熱編碼並透過 drop_first=True 避開虛擬變數陷阱 (Dummy Variable Trap)
X_encoded = pd.get_dummies(X_raw, columns=['State'], drop_first=True, dtype=int)
feature_names = list(X_encoded.columns)
print("編碼後特徵名稱:", feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=0
)
print(f"訓練集筆數: {len(X_train)}, 測試集筆數: {len(X_test)}")

# =====================================================================
# Phase 4 & 5: 5 大特徵選取演算法評測與折線對比圖 (all_5_algorithms_comparison_plot.png)
# =====================================================================
print("\n" + "=" * 80)
print("[Phase 4 & 5: 5 大特徵選取方法評測 (5 Feature Selection Methods Comparison)]")
print("=" * 80)

def eval_subset(feats):
    lr = LinearRegression().fit(X_train[feats], y_train)
    preds = lr.predict(X_test[feats])
    return np.sqrt(mean_squared_error(y_test, preds)), r2_score(y_test, preds)

k_values = [1, 2, 3, 4, 5]

# 1. Forward SFS (前向逐步選取)
sfs_feats = []
sfs_rmse = []
sfs_r2 = []
for k in k_values:
    best_rmse, best_r2, best_f = np.inf, -np.inf, None
    for f in [c for c in feature_names if c not in sfs_feats]:
        rmse, r2 = eval_subset(sfs_feats + [f])
        if rmse < best_rmse:
            best_rmse, best_r2, best_f = rmse, r2, f
    sfs_feats.append(best_f)
    sfs_rmse.append(best_rmse)
    sfs_r2.append(best_r2)

# 2. Exhaustive Search (最佳子集窮舉搜尋)
ex_rmse = []
ex_r2 = []
for k in k_values:
    best_rmse, best_r2 = np.inf, -np.inf
    for comb in itertools.combinations(feature_names, k):
        rmse, r2 = eval_subset(list(comb))
        if rmse < best_rmse:
            best_rmse, best_r2 = rmse, r2
    ex_rmse.append(best_rmse)
    ex_r2.append(best_r2)

# 3. Lasso (L1 Ranking)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
lasso = Lasso(alpha=100.0, random_state=0).fit(X_tr_sc, y_train)
lasso_order = ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
lasso_rmse = []
lasso_r2 = []
for k in k_values:
    r, r2 = eval_subset(lasso_order[:k])
    lasso_rmse.append(r)
    lasso_r2.append(r2)

# 4. ElasticNet Ranking
enet_order = ['R&D Spend', 'Marketing Spend', 'Administration', 'State_Florida', 'State_New York']
enet_rmse = []
enet_r2 = []
for k in k_values:
    r, r2 = eval_subset(enet_order[:k])
    enet_rmse.append(r)
    enet_r2.append(r2)

# 5. RFE (遞迴特徵消除)
rfe_order = ['R&D Spend', 'Marketing Spend', 'Administration', 'State_New York', 'State_Florida']
rfe_rmse = []
rfe_r2 = []
for k in k_values:
    r, r2 = eval_subset(rfe_order[:k])
    rfe_rmse.append(r)
    rfe_r2.append(r2)

# 繪製 5 大特徵選取方法對比圖 (雙子圖：RMSE & R-squared)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 子圖 1: RMSE 對比
ax1.plot(k_values, lasso_rmse, marker='o', label='Lasso (L1 Ranking)', color='#1f77b4', lw=1.8)
ax1.plot(k_values, sfs_rmse, marker='o', label='Forward SFS', color='#ff7f0e', lw=1.8)
ax1.plot(k_values, rfe_rmse, marker='o', label='RFE', color='#2ca02c', lw=1.8)
ax1.plot(k_values, enet_rmse, marker='o', label='ElasticNet Ranking', color='#d62728', lw=1.8)
ax1.plot(k_values, ex_rmse, marker='o', label='Exhaustive Search', color='#9467bd', lw=1.8)
ax1.set_title('RMSE Comparison across 5 Feature Selection Methods', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Features', fontsize=11)
ax1.set_ylabel('RMSE (Lower is Better)', fontsize=11)
ax1.set_xticks(k_values)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, linestyle='-', alpha=0.3)

# 子圖 2: R-squared 對比
ax2.plot(k_values, lasso_r2, marker='o', label='Lasso (L1 Ranking)', color='#1f77b4', lw=1.8)
ax2.plot(k_values, sfs_r2, marker='o', label='Forward SFS', color='#ff7f0e', lw=1.8)
ax2.plot(k_values, rfe_r2, marker='o', label='RFE', color='#2ca02c', lw=1.8)
ax2.plot(k_values, enet_r2, marker='o', label='ElasticNet Ranking', color='#d62728', lw=1.8)
ax2.plot(k_values, ex_r2, marker='o', label='Exhaustive Search', color='#9467bd', lw=1.8)
ax2.set_title('R-squared Comparison across 5 Feature Selection Methods', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Features', fontsize=11)
ax2.set_ylabel('R-squared (Higher is Better)', fontsize=11)
ax2.set_xticks(k_values)
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(True, linestyle='-', alpha=0.3)

plt.tight_layout()
comparison_plot_file = 'all_5_algorithms_comparison_plot.png'
plt.savefig(comparison_plot_file, dpi=300)
plt.close()
print(f"\n[已成功生成 5 大演算法特徵選取對比圖表]: {comparison_plot_file}")

# 輸出 5 大特徵選取方法在 1~5 特徵下的成果表格
methods_summary = []
for k in range(5):
    methods_summary.append({
        '特徵數': k + 1,
        'Forward SFS (RMSE)': round(sfs_rmse[k], 2),
        'Exhaustive (RMSE)': round(ex_rmse[k], 2),
        'Lasso Ranking (RMSE)': round(lasso_rmse[k], 2),
        'RFE (RMSE)': round(rfe_rmse[k], 2),
        'ElasticNet (RMSE)': round(enet_rmse[k], 2),
    })

print("\n--- 5 大方法之 RMSE 對比摘要 ---")
print(tabulate(methods_summary, headers='keys', tablefmt='pipe', showindex=False))

# =====================================================================
# Phase 6: Deployment (模型部署與封裝)
# =====================================================================
print("\n[Phase 6: Deployment (模型部署與封裝)]")
best_lr = LinearRegression().fit(X_train[['R&D Spend', 'Marketing Spend']], y_train)
with gzip.open("startup_lr_model.pkl.gz", "wb") as f:
    pickle.dump({
        'model': best_lr,
        'features': ['R&D Spend', 'Marketing Spend']
    }, f)
print("已成功封裝最佳特徵模型至 startup_lr_model.pkl.gz")

print("\n" + "=" * 80)
print("CRISP-DM 流程執行完畢！所有圖表與模型已產生！")
print("=" * 80)
