"""使用已訓練的決策樹，批次預測新案例。"""

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "DT_Results" / "best_model.joblib"

# 手動指定要預測的案例 CSV；每列代表一個案例。
INPUT_FILE = BASE_DIR / "datasets" / "new_cases.csv"
OUTPUT_FILE = BASE_DIR / "DT_Results" / "Predictions.csv"
TEMPLATE_FILE = BASE_DIR / "datasets" / "new_cases_template.csv"


if not MODEL_FILE.is_file():
    raise FileNotFoundError(
        f"找不到模型檔案：{MODEL_FILE}。請先執行 DT_all_in_one.py 訓練模型。"
    )

bundle = joblib.load(MODEL_FILE)
required_keys = {"model", "feature_columns", "positive_class"}
missing_keys = required_keys - bundle.keys()
if missing_keys:
    raise ValueError(f"模型檔案缺少必要資訊：{sorted(missing_keys)}")

cases = pd.read_csv(INPUT_FILE)
if cases.empty:
    raise ValueError("案例 CSV 沒有任何資料列。")

duplicate_columns = cases.columns[cases.columns.duplicated()].tolist()
if duplicate_columns:
    raise ValueError(f"案例 CSV 有重複欄位名稱：{duplicate_columns}")

model = bundle["model"]
feature_columns = bundle["feature_columns"]
positive_class = bundle["positive_class"]

if not INPUT_FILE.is_file():
    pd.DataFrame(columns=feature_columns).to_csv(
        TEMPLATE_FILE, index=False, encoding="utf-8-sig"
    )
    print(f"找不到案例 CSV：{INPUT_FILE}")
    print(f"已建立欄位範本：{TEMPLATE_FILE}")
    print("請填入一列或多列案例後另存為 new_cases.csv，再重新執行。")
    raise SystemExit(0)

missing_features = [
    column for column in feature_columns if column not in cases.columns
]
if missing_features:
    raise ValueError(
        "案例 CSV 缺少模型必要特徵欄位："
        f"{missing_features}"
    )

# 額外欄位（例如個案編號）不送入模型，但會保留在預測結果中。
model_input = cases.loc[:, feature_columns]
predictions = model.predict(model_input)

positive_index = list(model.classes_).index(positive_class)
positive_probabilities = model.predict_proba(model_input)[:, positive_index]

results = cases.copy()
results["Predicted_Class"] = predictions
results["Positive_Probability"] = positive_probabilities
results.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"已完成 {len(results)} 筆案例預測。")
print(f"結果輸出位置：{OUTPUT_FILE}")
print("\n預測結果：")
print(
    results[["Predicted_Class", "Positive_Probability"]]
    .assign(
        Positive_Probability=lambda frame: frame["Positive_Probability"].map(
            lambda value: f"{value:.1%}"
        )
    )
    .to_string(index=False)
)
