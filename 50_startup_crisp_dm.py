"""
AIoT-DA L6 Multiple Linear Regression (Startup 50 on Kaggle)
Following CRISP-DM Methodology
"""

import os
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 設定圖表字型避免亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("CRISP-DM 專案實作：50 Startups 多元線性回歸分析 (Multiple Linear Regression)")
print("=" * 70)

# =====================================================================
# Phase 1: Business Understanding (商業理解)
# =====================================================================
print("\n[Phase 1: Business Understanding (商業理解)]")
print("目標：預測新創公司的利潤 (Profit)，找出影響利潤最關鍵的支出項目 (R&D, Admin, Marketing)。")
print("利益關係人：企業投資人、高階主管、創業決策顧問。")
print("成功標準：高 R2 決定係數 (>= 0.90) 與低 RMSE / MAE 誤差。")

# =====================================================================
# Phase 2: Data Understanding (資料理解)
# =====================================================================
print("\n[Phase 2: Data Understanding (資料理解)]")
data_path = "50_Startups.csv"
if not os.path.exists(data_path):
    data_path = "50_Startups (2).csv"

print(f"載入資料檔案: {data_path}")
data = pd.read_csv(data_path)

print("\n--- 資料前 5 筆 (Head) ---")
print(data.head())

print("\n--- 資料欄位資訊 (Info) ---")
print(data.info())

print("\n--- 敘述統計 (Describe) ---")
print(data.describe())

print("\n--- 缺失值檢查 ---")
missing_vals = data.isnull().sum()
print(missing_vals)

# =====================================================================
# Phase 3: Data Preparation (資料準備)
# =====================================================================
print("\n[Phase 3: Data Preparation (資料準備)]")
# 1. 切分特徵 (X) 與目標變數 (y)
X_raw = data.drop('Profit', axis=1)
y = data['Profit']

# 2. 類別變數編碼 (One-Hot Encoding for 'State')
# 透過 drop_first=True 避開虛擬變數陷阱 (Dummy Variable Trap, 避免完全共線性)
X_encoded = pd.get_dummies(X_raw, columns=['State'], drop_first=True, dtype=int)
print("\n編碼後特徵欄位名稱:")
print(list(X_encoded.columns))

# 3. 切分訓練集與測試集 (80% 訓練, 20% 測試, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=0
)
print(f"訓練集樣本數: {X_train.shape[0]}, 測試集樣本數: {X_test.shape[0]}")

# =====================================================================
# Phase 4: Modeling (模型建立)
# =====================================================================
print("\n[Phase 4: Modeling (模型建立)]")
model = LinearRegression()
model.fit(X_train, y_train)

print(f"模型截距項 (Intercept, b0): {model.intercept_:.2f}")
print("模型回歸係數 (Coefficients):")
for col, coef in zip(X_encoded.columns, model.coef_):
    print(f"  - {col:25s}: {coef:12.4f}")

# =====================================================================
# Phase 5: Evaluation (模型評估)
# =====================================================================
print("\n[Phase 5: Evaluation (模型評估)]")

# 訓練集表現
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

# 測試集表現
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

print("測試集評估指標:")
print(f"  - R-squared (決定係數)      : {test_r2:.6f}")
print(f"  - Mean Absolute Error (MAE) : {mae:.2f}")
print(f"  - Mean Squared Error (MSE)  : {mse:.2f}")
print(f"  - Root Mean Squared Error   : {rmse:.2f}")

print("\n過擬合/欠擬合判斷 (Overfit / Underfit Check):")
print(f"  - 訓練集 R2: {train_r2:.4f}")
print(f"  - 測試集 R2: {test_r2:.4f}")
if abs(train_r2 - test_r2) < 0.05:
    print("  -> 模型表現穩健，訓練集與測試集準確率接近，無明顯過擬合。")
elif train_r2 > test_r2:
    print("  -> 訓練集表現顯著高於測試集，需注意輕微過擬合 (Overfitting)。")
else:
    print("  -> 測試集表現優異，模型泛化良好。")

# 視覺化：實際值 vs 預測值
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='#2563EB', edgecolors='k', s=80, alpha=0.8, label='Test Data')
# 理想對角線
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit Line')
plt.xlabel('Actual Profit (實際利潤)', fontsize=12)
plt.ylabel('Predicted Profit (預測利潤)', fontsize=12)
plt.title('Actual vs Predicted Profit (50 Startups)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
scatter_plot_path = "actual_vs_predicted.png"
plt.savefig(scatter_plot_path, dpi=300)
plt.close()
print(f"\n[視覺化圖表已儲存]: {scatter_plot_path}")

# --- Forward Feature Selection (前向特徵選取實驗) ---
print("\n" + "=" * 50)
print("特徵選取優化實驗：Forward Feature Selection (前向逐步選取)")
print("=" * 50)

def evaluate_performance(y_true, y_pred):
    current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    current_r2 = r2_score(y_true, y_pred)
    return current_rmse, current_r2

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
print("\nForward Selection 成果表格 (Markdown Format):")
print(tabulate(forward_results_df, headers='keys', tablefmt='pipe', showindex=False))

# =====================================================================
# Phase 6: Deployment (模型部署)
# =====================================================================
print("\n[Phase 6: Deployment (模型部署與封裝)]")
model_deployment_path = "startup_lr_model.pkl.gz"
with gzip.open(model_deployment_path, "wb") as f:
    pickle.dump({
        'model': model,
        'feature_names': list(X_encoded.columns)
    }, f)
print(f"已使用 gzip + pickle 封裝模型至: {model_deployment_path}")

# 示範載入並進行新樣本預測
with gzip.open(model_deployment_path, "rb") as f:
    loaded_package = pickle.load(f)
    loaded_model = loaded_package['model']
    features = loaded_package['feature_names']

sample_new_data = pd.DataFrame([{
    'R&D Spend': 160000.0,
    'Administration': 130000.0,
    'Marketing Spend': 300000.0,
    'State_Florida': 1,
    'State_New York': 0
}], columns=features)

predicted_sample_profit = loaded_model.predict(sample_new_data)[0]
print(f"範例新創公司預測利潤: ${predicted_sample_profit:,.2f}")
print("\n" + "=" * 70)
print("CRISP-DM 流程執行完畢！")
print("=" * 70)
