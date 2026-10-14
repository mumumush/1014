---
name: crisp-dm-stock-autoregression
description: >-
  Auto-execute CRISP-DM workflow for Stock Price Prediction (2330.TW) using Auto Regression,
  lagged features, moving averages, and time-series evaluation.
---

# CRISP-DM Stock Price Prediction Workflow (2330.TW)

This skill provides an automated, standardized end-to-end CRISP-DM workflow to predict stock closing prices using Auto Regression (自迴歸模型).

## Workflow Execution Steps

### 1. Phase 1: Business Understanding (商業理解)
- **Objective**: Predict the next-day closing price of TSMC (`2330.TW`) to support quantitative trading and risk assessment.
- **Success Criteria**: Test $R^2 \ge 0.90$, RMSE $< 25$ NTD on testing period.

### 2. Phase 2: Data Understanding (資料獲取與理解)
- Fetch historical daily stock data (`Open`, `High`, `Low`, `Close`, `Volume`) via Yahoo Finance (`yfinance`) or local CSV (`2330_TW.csv`).
- Verify data continuity, date range, and null values.

### 3. Phase 3: Data Preparation (時序特徵工程)
- **Lag Features**: Shift close price by 1, 2, 3, and 5 trading days (`Close_lag1`, `Close_lag2`, `Close_lag3`, `Close_lag5`).
- **Moving Averages**: 5-day moving average (`MA5`) and 20-day moving average (`MA20`).
- **Time-Series Split**: Strictly split 80% training and 20% testing by chronological order (no random shuffle).

### 4. Phase 4: Modeling (自迴歸建模)
- Train an Auto-Regressive Ridge / Linear model:
  $$\hat{P}_t = b_0 + w_1 P_{t-1} + w_2 P_{t-2} + w_3 P_{t-3} + w_5 P_{t-5} + w_{MA5} \text{MA5}_t + w_{MA20} \text{MA20}_t$$

### 5. Phase 5: Evaluation (模型評估與視覺化)
- Compute test metrics: $R^2$, RMSE, MAE.
- Plot historical vs. predicted trend curve and save to `tsmc_2330_stock_prediction.png`.

### 6. Phase 6: Deployment (次日推論與模型封裝)
- Feed latest trading day features to forecast the next trading day's closing price.
- Package model to `tsmc_2330_ar_model.pkl.gz`.
