# Desition-Tree
訓練model、預測

輸入各項特徵數值(x/Features)，輸出預測結果(y/TARGET)
可將特徵文字轉數值

**一、訓練:**

  1.將要訓練的資料(.csv，可從https://www.kaggle.com 找)放到dataset資料夾

  2.打開DT_all_in_one.py

  3.修改以下

    (1. 基本設定，其他有需也可改)
  
    DATA_FILE最後"ParkinsonsDNN.csv"改成你的資料檔名
  
    TARGET = "status"改成你的目標欄位名(要判斷的答案，通常在最後一欄)
  
    POSITIVE_CLASS = 1 改成目標欄位正類的值(如陽性)

  
    (2. 讀取資料與前處理)
  
    if col.lower() in [TARGET, "name"] 可增刪排除的欄位

    (4. Grid Search 設定，參數有需也可改)

4.執行，獲得模型與相關結果於DT_Results資料夾 (best_model.joblib為模型)

**二、預測:**

1.將要預測的資料(.csv)放到dataset資料夾(可不含TARGET)

2.手動指定INPUT_FILE最後"new_cases.csv"改成你的資料檔名 (或把檔名改成new_cases.csv)

3.執行，獲得結果於DT_Results資料夾 (Predictions.csv為預測結果)
