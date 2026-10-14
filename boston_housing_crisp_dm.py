"""
AIoT-DA L6: Kaggle Boston Housing Price Prediction using Multiple Linear Regression & Machine Learning
Following CRISP-DM Methodology Workflow (Identical to 50 Startups Workflow)
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
print("CRISP-DM 專案實作：Kaggle Boston Housing 房價預測與 5 大演算法橫向評測")
print("=" * 80)

# =====================================================================
# Phase 1: Business Understanding (商業理解)
# =====================================================================
print("\n[Phase 1: Business Understanding (商業理解)]")
print("目標：預測波士頓地區住宅自住房屋價格中位數 (MEDV，單位：千美元)。")
print("評估各項社區特徵（犯罪率 CRIM、平均房間數 RM、低收入人口比例 LSTAT、學生教師比 PTRATIO）之影響力。")
print("成功標準：高決定係數 R2 與低 RMSE 誤差。")

# =====================================================================
# Phase 2: Data Understanding (資料理解)
# =====================================================================
print("\n[Phase 2: Data Understanding (資料理解)]")
csv_path = "boston_housing.csv"
if not os.path.exists(csv_path):
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    df.to_csv(csv_path, index=False)
else:
    df = pd.read_csv(csv_path)

print(f"載入資料檔案: {csv_path} (資料維度: {df.shape})")
print("\n--- 資料前 5 筆 ---")
print(df.head())
print("\n--- 缺失值檢查 ---")
print(df.isnull().sum())

# =====================================================================
# Phase 3: Data Preparation (資料準備)
# =====================================================================
print("\n[Phase 3: Data Preparation (資料準備)]")
X = df.drop('medv', axis=1)
y = df['medv']
feature_names = list(X.columns)

# 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print(f"特徵數量: {len(feature_names)} 欄位")
print(f"訓練集樣本數: {len(X_train)}, 測試集樣本數: {len(X_test)}")

# =====================================================================
# Phase 4: Modeling & Benchmark of 5 Regression Algorithms
# =====================================================================
print("\n" + "=" * 80)
print("[Phase 4: 5 大回歸演算法評測排行榜 (5 Regression Algorithms Benchmark)]")
print("=" * 80)

algorithms = {
    '梯度提升 (Gradient Boosting)': GradientBoostingRegressor(n_estimators=100, random_state=0),
    '隨機森林 (Random Forest)': RandomForestRegressor(n_estimators=100, random_state=0),
    '多元線性回歸 (Linear Regression, OLS)': LinearRegression(),
    '嶺回歸 (Ridge Regression, L2)': Ridge(alpha=1.0),
    '套索回歸 (Lasso Regression, L1)': Lasso(alpha=0.1, random_state=0)
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
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring='r2').mean()
    
    benchmark_records.append({
        '演算法 (Algorithm)': name,
        '測試集 R2': round(test_r2, 6),
        '測試集 RMSE ($k)': round(test_rmse, 3),
        '測試集 MAE ($k)': round(test_mae, 3),
        '5-Fold CV R2': round(cv_r2, 6)
    })

benchmark_df = pd.DataFrame(benchmark_records).sort_values(by='測試集 R2', ascending=False)
print("\n" + tabulate(benchmark_df, headers='keys', tablefmt='pipe', showindex=False))

# =====================================================================
# Phase 5: 5 大特徵選取方法評測與對比圖 (boston_all_5_algorithms_comparison_plot.png)
# =====================================================================
print("\n" + "=" * 80)
print("[Phase 5: 5 大特徵選取方法評測 (5 Feature Selection Methods Comparison)]")
print("=" * 80)

def eval_subset(feats):
    lr = LinearRegression().fit(X_train[feats], y_train)
    preds = lr.predict(X_test[feats])
    return np.sqrt(mean_squared_error(y_test, preds)), r2_score(y_test, preds)

k_values = [1, 2, 3, 4, 5]

# 1. Forward SFS (前向逐步挑選)
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

# 2. Exhaustive Search (最佳子集窮舉挑選)
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

# 3. Lasso Ranking
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
lasso = Lasso(alpha=0.1, random_state=0).fit(X_tr_sc, y_train)
lasso_order = [feature_names[i] for i in np.argsort(-np.abs(lasso.coef_))]
lasso_rmse = [eval_subset(lasso_order[:k])[0] for k in k_values]
lasso_r2 = [eval_subset(lasso_order[:k])[1] for k in k_values]

# 4. ElasticNet Ranking
enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=0).fit(X_tr_sc, y_train)
enet_order = [feature_names[i] for i in np.argsort(-np.abs(enet.coef_))]
enet_rmse = [eval_subset(enet_order[:k])[0] for k in k_values]
enet_r2 = [eval_subset(enet_order[:k])[1] for k in k_values]

# 5. RFE
rfe_model = RFE(estimator=LinearRegression(), n_features_to_select=1).fit(X_tr_sc, y_train)
rfe_order = [feature_names[i] for i in np.argsort(rfe_model.ranking_)]
rfe_rmse = [eval_subset(rfe_order[:k])[0] for k in k_values]
rfe_r2 = [eval_subset(rfe_order[:k])[1] for k in k_values]

# 繪製雙子圖折線圖 (完全比照 50 Startups)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(k_values, lasso_rmse, marker='o', label='Lasso (L1 Ranking)', color='#1f77b4', lw=1.8)
ax1.plot(k_values, sfs_rmse, marker='o', label='Forward SFS', color='#ff7f0e', lw=1.8)
ax1.plot(k_values, rfe_rmse, marker='o', label='RFE', color='#2ca02c', lw=1.8)
ax1.plot(k_values, enet_rmse, marker='o', label='ElasticNet Ranking', color='#d62728', lw=1.8)
ax1.plot(k_values, ex_rmse, marker='o', label='Exhaustive Search', color='#9467bd', lw=1.8)
ax1.set_title('RMSE Comparison across 5 Feature Selection Methods (Boston)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Features', fontsize=11)
ax1.set_ylabel('RMSE (Lower is Better)', fontsize=11)
ax1.set_xticks(k_values)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, linestyle='-', alpha=0.3)

ax2.plot(k_values, lasso_r2, marker='o', label='Lasso (L1 Ranking)', color='#1f77b4', lw=1.8)
ax2.plot(k_values, sfs_r2, marker='o', label='Forward SFS', color='#ff7f0e', lw=1.8)
ax2.plot(k_values, rfe_r2, marker='o', label='RFE', color='#2ca02c', lw=1.8)
ax2.plot(k_values, enet_r2, marker='o', label='ElasticNet Ranking', color='#d62728', lw=1.8)
ax2.plot(k_values, ex_r2, marker='o', label='Exhaustive Search', color='#9467bd', lw=1.8)
ax2.set_title('R-squared Comparison across 5 Feature Selection Methods (Boston)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Features', fontsize=11)
ax2.set_ylabel('R-squared (Higher is Better)', fontsize=11)
ax2.set_xticks(k_values)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, linestyle='-', alpha=0.3)

plt.tight_layout()
plot_file = 'boston_all_5_algorithms_comparison_plot.png'
plt.savefig(plot_file, dpi=300)
plt.close()
print(f"\n[已生成 5 大特徵選取對比圖表]: {plot_file}")

# 繪製 Actual vs Predicted
gb_pred = fitted_models['梯度提升 (Gradient Boosting)'].predict(X_test)
lr_pred = fitted_models['多元線性回歸 (Linear Regression, OLS)'].predict(X_test)

plt.figure(figsize=(9, 6))
plt.scatter(y_test, lr_pred, color='#3B82F6', s=60, alpha=0.7, edgecolors='k', label='Linear Regression (R2=0.5892)')
plt.scatter(y_test, gb_pred, color='#10B981', s=70, marker='^', alpha=0.85, edgecolors='k', label='Gradient Boosting (R2=0.7809)')
min_v, max_v = min(y_test.min(), min(lr_pred.min(), gb_pred.min())), max(y_test.max(), max(lr_pred.max(), gb_pred.max()))
plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Fit Line (y=x)')
plt.xlabel('Actual MEDV (實際房價 $1000s)', fontsize=12)
plt.ylabel('Predicted MEDV (預測房價 $1000s)', fontsize=12)
plt.title('Boston Housing: Actual vs Predicted Price', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("boston_actual_vs_predicted.png", dpi=300)
plt.close()
print("[已生成實際 vs 預測對比圖]: boston_actual_vs_predicted.png")

# =====================================================================
# Phase 6: Deployment (模型部署與封裝)
# =====================================================================
print("\n[Phase 6: Deployment (模型部署與封裝)]")
with gzip.open("boston_housing_model.pkl.gz", "wb") as f:
    pickle.dump({
        'models': fitted_models,
        'feature_names': feature_names
    }, f)
print("已成功封裝波士頓房價模型至 boston_housing_model.pkl.gz")
print("=" * 80)
