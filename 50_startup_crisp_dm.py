"""
AIoT-DA L6 Multiple Linear Regression & 5 Regression Algorithms Benchmark (50 Startups)
Following CRISP-DM Methodology
"""

import os
import sys
import gzip
import pickle
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 設定圖表中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 75)
print("CRISP-DM 專案實作：50 Startups 回歸分析與 5 大最佳機器學習演算法評測")
print("=" * 75)

# =====================================================================
# Phase 1: Business Understanding (商業理解)
# =====================================================================
print("\n[Phase 1: Business Understanding (商業理解)]")
print("目標：預測新創公司的利潤 (Profit)，評估 R&D、行政、行銷支出及地理州別之影響力。")
print("評測 5 大經典回歸演算法（Linear, Ridge, Lasso, Random Forest, Gradient Boosting），找出預測精度最高之模型。")
print("成功標準：高決定係數 R2 (>= 0.90) 與低 RMSE/MAE 誤差。")

# =====================================================================
# Phase 2: Data Understanding (資料理解)
# =====================================================================
print("\n[Phase 2: Data Understanding (資料理解)]")
data_path = "50_Startups.csv" if os.path.exists("50_Startups.csv") else "50_Startups (2).csv"
data = pd.read_csv(data_path)

print(f"載入資料檔案: {data_path} (形狀: {data.shape})")
print("\n--- 資料前 5 筆 ---")
print(data.head())
print("\n--- 缺失值檢查 ---")
print(data.isnull().sum())

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

# 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=0
)
print(f"訓練集樣本數: {len(X_train)}, 測試集樣本數: {len(X_test)}")

# =====================================================================
# Phase 4 & 5: Modeling & Benchmark of 5 Best Algorithms (5 大演算法 PK)
# =====================================================================
print("\n" + "=" * 75)
print("[Phase 4 & 5: 5 大最佳回歸演算法評測排行榜 (5 Regression Algorithms Benchmark)]")
print("=" * 75)

algorithms = {
    '隨機森林 (Random Forest)': RandomForestRegressor(n_estimators=100, random_state=0),
    '梯度提升 (Gradient Boosting)': GradientBoostingRegressor(n_estimators=100, random_state=0),
    '套索回歸 (Lasso Regression, L1)': Lasso(alpha=100.0, random_state=0),
    '嶺回歸 (Ridge Regression, L2)': Ridge(alpha=1.0),
    '多元線性回歸 (Linear Regression, OLS)': LinearRegression()
}

kf = KFold(n_splits=5, shuffle=True, random_state=0)
benchmark_records = []
fitted_models = {}

for name, model in algorithms.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model
    
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    cv_r2 = cross_val_score(model, X_encoded, y, cv=kf, scoring='r2').mean()
    
    benchmark_records.append({
        '演算法 (Algorithm)': name,
        '測試集 R2': round(test_r2, 6),
        '測試集 RMSE': round(test_rmse, 2),
        '測試集 MAE': round(test_mae, 2),
        '5-Fold CV R2': round(cv_r2, 6)
    })

benchmark_df = pd.DataFrame(benchmark_records).sort_values(by='測試集 R2', ascending=False)
print("\n" + tabulate(benchmark_df, headers='keys', tablefmt='pipe', showindex=False))

# 最佳模型 (Random Forest)
best_model_name = benchmark_df.iloc[0]['演算法 (Algorithm)']
print(f"\n🏆 全域表現最佳演算法：{best_model_name}")

# =====================================================================
# Phase 5: Forward Feature Selection (前向逐步特徵選取實驗)
# =====================================================================
print("\n" + "=" * 75)
print("特徵選取深入探討：Forward Feature Selection (1 到 5 個特徵表現)")
print("=" * 75)

def evaluate_performance(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)

def forward_selection(X_tr, X_te, y_tr, y_te, max_features):
    selected_features = []
    results = []
    all_features = list(X_tr.columns)

    for i in range(1, max_features + 1):
        remaining = [f for f in all_features if f not in selected_features]
        best_rmse = np.inf
        best_r2 = -np.inf
        best_feature = None

        for feature in remaining:
            features_to_use = selected_features + [feature]
            reg = LinearRegression()
            reg.fit(X_tr[features_to_use], y_tr)
            preds = reg.predict(X_te[features_to_use])
            cur_rmse, cur_r2 = evaluate_performance(y_te, preds)

            if cur_rmse < best_rmse:
                best_rmse = cur_rmse
                best_r2 = cur_r2
                best_feature = feature

        selected_features.append(best_feature)
        results.append({
            'Number of Features': i,
            'Selected Features': ", ".join(selected_features),
            'RMSE': round(best_rmse, 6),
            'R-squared': round(best_r2, 6)
        })

    return pd.DataFrame(results)

forward_results_df = forward_selection(X_train, X_test, y_train, y_test, max_features=len(X_train.columns))
print("\nForward Selection 成果表格:")
print(tabulate(forward_results_df, headers='keys', tablefmt='pipe', showindex=False))

# 繪製 Actual vs Predicted (以 Random Forest 與 Linear Regression 對比)
lr_pred = fitted_models['多元線性回歸 (Linear Regression, OLS)'].predict(X_test)
rf_pred = fitted_models['隨機森林 (Random Forest)'].predict(X_test)

plt.figure(figsize=(9, 6))
plt.scatter(y_test, lr_pred, color='#3B82F6', s=80, alpha=0.8, edgecolors='k', label='Linear Regression (R2=0.9347)')
plt.scatter(y_test, rf_pred, color='#10B981', s=90, marker='^', alpha=0.9, edgecolors='k', label='Random Forest (R2=0.9699)')
min_val, max_val = min(y_test.min(), min(lr_pred.min(), rf_pred.min())), max(y_test.max(), max(lr_pred.max(), rf_pred.max()))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit Line (y=x)')
plt.xlabel('Actual Profit (實際利潤)', fontsize=12)
plt.ylabel('Predicted Profit (預測利潤)', fontsize=12)
plt.title('Actual vs Predicted Profit: Linear Regression vs Random Forest', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.close()
print("\n[視覺化圖表已更新儲存]: actual_vs_predicted.png")

# =====================================================================
# Phase 6: Deployment (模型封裝與多模型推論)
# =====================================================================
print("\n[Phase 6: Deployment (模型部署與封裝)]")
model_deployment_path = "startup_lr_model.pkl.gz"
with gzip.open(model_deployment_path, "wb") as f:
    pickle.dump({
        'models': fitted_models,
        'feature_names': feature_names
    }, f)
print(f"已使用 gzip + pickle 封裝 5 大模型至: {model_deployment_path}")

sample_new_data = pd.DataFrame([{
    'R&D Spend': 160000.0,
    'Administration': 130000.0,
    'Marketing Spend': 300000.0,
    'State_Florida': 1,
    'State_New York': 0
}], columns=feature_names)

print("\n--- 針對範例新創公司之 5 大演算法推論利潤 ---")
for m_name, mdl in fitted_models.items():
    pred_val = mdl.predict(sample_new_data)[0]
    print(f"  - {m_name:38s}: ${pred_val:,.2f}")

print("\n" + "=" * 75)
print("CRISP-DM 完整流程與 5 大演算法評測執行完畢！")
print("=" * 75)
