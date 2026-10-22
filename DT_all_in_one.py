#自動判斷二元或多元分類

#Decision Tree：全特徵模型 vs 前五重要特徵模型
# 功能：
# 1. 載入 diabetes012.csv，進行資料檢查與前處理
# 2. 分割訓練集與測試集
# 3. 使用全特徵建立決策樹，透過 Grid Search 與 5 折交叉驗證尋找最佳參數組合
# 4. 根據全特徵最佳模型的 Feature Importance，排序並顯示所有特徵的重要性
# 5. 建立前五特徵模型，透過 Pipeline 在交叉驗證的各訓練折內自動選取五個特徵，並進行 Grid Search
# 6. 比較兩個模型的交叉驗證平均 F1 與測試集表現：Accuracy、Precision、Recall、F1-score、ROC-AUC、AP
# 7. 顯示兩個模型的混淆矩陣與分類報告
# 8. 依交叉驗證平均 F1 選出最佳模型
# 9. 輸出最佳模型的 ROC 曲線、PRC 曲線、決策樹圖與文字規則
# 10. 輸出全特徵模型的特徵重要性圖與執行摘要
# ============================================================

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定中文字型（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

# 避免負號顯示異常
plt.rcParams['axes.unicode_minus'] = False

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    make_scorer
)

# ============================================================
# 1. 基本設定 check
# ============================================================

start_time = time.time()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "datasets" / "ParkinsonsDNN.csvv"
OUTPUT_DIR = BASE_DIR / "DT_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.3
RANDOM_STATE = 42
CV_FOLDS = 5
TOP_N_FEATURES = 5

TARGET = "status" #要判斷的結果

# TARGET如為二元，正類設為TARGET欄位中的其中一個值
POSITIVE_CLASS = "1"


# ============================================================
# 2. 讀取資料與前處理 check
# ============================================================

df = pd.read_csv(DATA_FILE)

if TARGET not in df.columns:
    raise ValueError(f"找不到目標欄位：{TARGET}")

# 排除目標欄位、姓名及多餘的索引欄
exclude_cols = [
    col for col in df.columns
    if col in [TARGET, "name", "Student_ID"]
    or col.lower().startswith("unnamed:")
]

X = df.drop(columns=exclude_cols)
y = df[TARGET]

# One-Hot Encoding將類別型文字轉成數值
X = pd.get_dummies(X, dtype=int)

# 將所有布林型態的特徵轉成整數
bool_cols = X.select_dtypes(include=["bool"]).columns
X[bool_cols] = X[bool_cols].astype(int)

# 確認所有特徵都是數值型態
non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()

if non_numeric:
    raise ValueError(
        f"以下特徵不是數值型態，請先處理：{non_numeric}"
    )

all_missing_features = X.columns[X.isnull().all()].tolist()
if all_missing_features:
    raise ValueError(
        "以下特徵全部都是缺失值，無法以中位數補值："
        f"{all_missing_features}"
    )

# 目標欄位不能可靠地補值；特徵缺失值則由 Pipeline 在每個訓練 fold
# 以中位數補值，避免交叉驗證資料洩漏。
if y.isnull().any():
    raise ValueError(f"目標欄位 {TARGET!r} 包含缺失值，無法訓練分類模型。")

# 自動取得所有類別，支援二元與三類以上的多元分類。
class_order = sorted(pd.unique(y).tolist())
n_classes = len(class_order)

if n_classes < 2:
    raise ValueError(
        f"目標欄位 {TARGET!r} 至少需要兩個類別，目前為：{class_order}"
    )

# POSITIVE_CLASS 僅用於二元分類的 ROC / PRC。
# 若二元資料的標籤不是預設的 1，便自動使用排序後的最後一類。
if n_classes == 2 and POSITIVE_CLASS not in class_order:
    POSITIVE_CLASS = class_order[-1]
    print(f"指定正類不存在，二元分類改用 {POSITIVE_CLASS!r} 作為正類。")

if n_classes == 2:
    # 二元分類：將正類固定放在第二位。
    class_order = [
        label for label in class_order if label != POSITIVE_CLASS
    ] + [POSITIVE_CLASS]

if y.value_counts().min() < CV_FOLDS:
    raise ValueError(
        f"少數類別的樣本數不足 {CV_FOLDS}，無法使用目前的交叉驗證設定。"
    )


print("=" * 60)
print("資料基本資訊")
print("=" * 60)
print("資料筆數：", len(df))
print("特徵數量：", X.shape[1])
print("\n類別分布：")
print(y.value_counts().sort_index())
if X.isnull().any().any():
    print("\n偵測到特徵缺失值：將在每個交叉驗證訓練 fold 以中位數補值。")


# ============================================================
# 3. 分割訓練集與測試集
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

n_samples = len(X_train)
print("\n訓練集筆數：", n_samples)
print("測試集筆數：", len(X_test))

if y_train.value_counts().min() < CV_FOLDS:
    raise ValueError(
        f"少數類別的樣本數不足 {CV_FOLDS}，無法使用目前的交叉驗證設定。"
    )
    
# ============================================================
# 4. Grid Search 設定+ 交叉驗證
# ============================================================
leaf_ratios = [0.005, 0.01, 0.02, 0.05, 0.10]
split_ratios = [0.01, 0.02, 0.05, 0.10, 0.15]
min_samples_leaf_list = sorted(
    list({max(2, int(n_samples * r)) for r in leaf_ratios})
)
min_samples_split_list = sorted(
    list({max(4, int(n_samples * r)) for r in split_ratios})
)

tree_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": min_samples_leaf_list,
    "min_samples_split": min_samples_split_list,
    "max_features": [None, "sqrt"],
    "ccp_alpha": [0.0, 0.001, 0.005, 0.01]
}

cv_strategy = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


def prefixed_tree_param_grid():
    """將決策樹參數套用到 Pipeline 的 tree step。"""
    return {f"tree__{name}": values for name, values in tree_param_grid.items()}


def build_model_pipeline(select_top_features=False):
    """建立不會在 CV fold 間洩漏補值或特徵選擇資訊的 Pipeline。"""
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if select_top_features:
        steps.append((
            "top5_selector",
            SelectFromModel(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                threshold=-np.inf,
                max_features=TOP_N_FEATURES
            )
        ))

    steps.append(("tree", DecisionTreeClassifier(random_state=RANDOM_STATE)))
    return Pipeline(steps)


def run_grid_search(X_tr, y_tr, model_name, estimator):
    """Grid Search + 5 折交叉驗證，輸出四項指標的各折分數與平均值。"""

    print("\n" + "=" * 70)
    print(f"{model_name}：開始 Grid Search + 5 折交叉驗證")
    print("=" * 70)

    # Macro 平均讓每個類別權重相同，同時支援二元與多元分類。
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "recall": make_scorer(
            recall_score, average="macro", zero_division=0
        ),
        "f1": make_scorer(
            f1_score, average="macro", zero_division=0
        )
    }

    search = GridSearchCV(
        estimator=estimator,
        param_grid=prefixed_tree_param_grid(),
        scoring=scoring,
        refit="f1",  # 依五折平均 f1 選最佳參數
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=True
    )

    search.fit(X_tr, y_tr)

    print("\n最佳參數：")
    print(search.best_params_)

    best_index = search.best_index_

    # 顯示最佳參數組合在每一折的四項指標
    fold_results = []

    for i in range(CV_FOLDS):
        fold_results.append({
            "Fold": f"第 {i + 1} 折",
            "Precision": search.cv_results_[
                f"split{i}_test_precision"
            ][best_index],
            "Recall": search.cv_results_[
                f"split{i}_test_recall"
            ][best_index],
            "F1-score": search.cv_results_[
                f"split{i}_test_f1"
            ][best_index],
            "Accuracy": search.cv_results_[
                f"split{i}_test_accuracy"
            ][best_index]
        })

    # 計算五折平均
    average_result = {
        "Fold": "五折平均",
        "Precision": search.cv_results_[
            "mean_test_precision"
        ][best_index],
        "Recall": search.cv_results_[
            "mean_test_recall"
        ][best_index],
        "F1-score": search.cv_results_[
            "mean_test_f1"
        ][best_index],
        "Accuracy": search.cv_results_[
            "mean_test_accuracy"
        ][best_index]
    }

    fold_results.append(average_result)

    result_df = pd.DataFrame(fold_results)

    print("\n最佳參數組合的交叉驗證結果：")
    print(result_df.round(4).to_string(index=False))

    print(
        f"\nGridSearchCV 最佳五折平均f1："
        f"{search.best_score_:.4f}"
    )

    # 回傳搜尋結果
    return search

# ============================================================
# 5. 模型一：使用全部特徵
# ============================================================

all_search = run_grid_search(
    X_train,
    y_train,
    "全特徵模型",
    build_model_pipeline()
)

all_model = all_search.best_estimator_
all_tree_model = all_model.named_steps["tree"]

# ============================================================
# 6. 印出全部特徵排名
# ============================================================

importance_df = pd.DataFrame({
    "Index": [
        X.columns.get_loc(col)
        for col in X_train.columns
    ],
    "Feature": X_train.columns,
    "Importance": all_tree_model.feature_importances_
})

# 依重要性由高到低排序
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# 輸出全部特徵排名（包含原始欄位編號）
print("\n" + "=" * 55)
print("特徵重要性排名")
print("=" * 55)

print(
    importance_df.to_string(
        index=False,
        formatters={
            "Importance": "{:.6f}".format
        }
    )
)

# ============================================================
# 7. 模型二：在每個 CV fold 內自動選出前五重要特徵 + 印出特徵排名
# ============================================================

top5_search = run_grid_search(
    X_train,
    y_train,
    "前五重要特徵模型",
    build_model_pipeline(select_top_features=True)
)

top5_model = top5_search.best_estimator_
top5_tree_model = top5_model.named_steps["tree"]
top5_support = top5_model.named_steps["top5_selector"].get_support()
top5_features = X_train.columns[top5_support].tolist()

print("\nTop 5 模型最終選用的特徵：")
print(top5_features)

#印出特徵排名
# 前五特徵 Pipeline 的選擇器
selector = top5_model.named_steps["top5_selector"]

# 取得選擇器內部決策樹對所有特徵的評分
selector_importance = selector.estimator_.feature_importances_

selector_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": selector_importance
}).sort_values("Importance", ascending=False)

print("\n前五特徵選擇器：全部特徵排名")
print(selector_importance_df.to_string(index=False))


# ============================================================
# 8. 評估兩個模型
# ============================================================

def evaluate_model(model, X_te, y_te, model_name):
    """評估二元或多元分類，並回傳比較表與 ROC / PRC 所需資料。"""
    from sklearn.preprocessing import label_binarize

    pred = model.predict(X_te)
    classes = list(model.classes_)
    pred_prob_all = model.predict_proba(X_te)

    accuracy = accuracy_score(y_te, pred)
    precision_score_value = precision_score(
        y_te, pred, average="macro", zero_division=0
    )
    recall_score_value = recall_score(
        y_te, pred, average="macro", zero_division=0
    )
    f1_score_value = f1_score(
        y_te, pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_te, pred, labels=class_order)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(f"Accuracy        ：{accuracy:.4f}")
    print(f"Precision Macro ：{precision_score_value:.4f}")
    print(f"Recall Macro    ：{recall_score_value:.4f}")
    print(f"F1 Macro        ：{f1_score_value:.4f}")

    print("\nConfusion Matrix：")
    print(pd.DataFrame(
        cm,
        index=[f"實際 {label}" for label in class_order],
        columns=[f"預測 {label}" for label in class_order]
    ))

    print("\nClassification Report：")
    print(classification_report(
        y_te,
        pred,
        labels=class_order,
        target_names=[str(label) for label in class_order],
        zero_division=0
    ))

    if len(classes) == 2:
        # 二元分類：以指定正類計算 ROC-AUC、AP 與曲線。
        positive_index = classes.index(POSITIVE_CLASS)
        pred_prob = pred_prob_all[:, positive_index]
        y_binary = (np.asarray(y_te) == POSITIVE_CLASS).astype(int)

        auc = roc_auc_score(y_binary, pred_prob)
        ap = average_precision_score(y_binary, pred_prob)
        fpr, tpr, _ = roc_curve(y_binary, pred_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_binary, pred_prob
        )
        auc_curve = auc
        ap_curve = ap
        curve_description = f"正類 {POSITIVE_CLASS!r}"

    else:
        # 多元分類：整體指標採 Macro OvR；圖表採 Micro-average 曲線。
        y_binary = label_binarize(y_te, classes=classes)

        auc = roc_auc_score(
            y_te,
            pred_prob_all,
            labels=classes,
            multi_class="ovr",
            average="macro"
        )
        ap = average_precision_score(
            y_binary, pred_prob_all, average="macro"
        )

        fpr, tpr, _ = roc_curve(
            y_binary.ravel(), pred_prob_all.ravel()
        )
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_binary.ravel(), pred_prob_all.ravel()
        )
        auc_curve = roc_auc_score(
            y_binary.ravel(), pred_prob_all.ravel()
        )
        ap_curve = average_precision_score(
            y_binary, pred_prob_all, average="micro"
        )
        curve_description = "Micro-average"

    print(f"ROC-AUC (整體) ：{auc:.4f}")
    print(f"AP (整體)      ：{ap:.4f}")
    if len(classes) > 2:
        print("多元分類 ROC / PRC 圖表使用 Micro-average 曲線。")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision_score_value,
        "Recall": recall_score_value,
        "F1": f1_score_value,
        "ROC_AUC": auc,
        "AP": ap,
        "ROC_AUC_Curve": auc_curve,
        "AP_Curve": ap_curve,
        "Curve_Description": curve_description,
        "Confusion_Matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision_curve,
        "recall": recall_curve,
        "pred": pred,
        "pred_prob": pred_prob_all
    }


# 評估全特徵模型
all_result = evaluate_model(
    all_model,
    X_test,
    y_test,
    "全特徵模型"
)

# 評估前五特徵模型
top5_result = evaluate_model(
    top5_model,
    X_test,
    y_test,
    "前五重要特徵模型"
)


# ============================================================
# 9. 模型比較表：二元與多元分類皆使用 Macro 指標
# ============================================================

comparison_df = pd.DataFrame([
    {
        "Model": "全特徵",
        "Feature_Count": X_train.shape[1],
        "CV_Mean_F1": all_search.best_score_,
        "Test_Accuracy": all_result["Accuracy"],
        "Test_Precision": all_result["Precision"],
        "Test_Recall": all_result["Recall"],
        "Test_F1": all_result["F1"],
        "Test_ROC_AUC": all_result["ROC_AUC"],
        "Test_AP": all_result["AP"]
    },
    {
        "Model": "前五重要特徵",
        "Feature_Count": len(top5_features),
        "CV_Mean_F1": top5_search.best_score_,
        "Test_Accuracy": top5_result["Accuracy"],
        "Test_Precision": top5_result["Precision"],
        "Test_Recall": top5_result["Recall"],
        "Test_F1": top5_result["F1"],
        "Test_ROC_AUC": top5_result["ROC_AUC"],
        "Test_AP": top5_result["AP"]
    }
])

print("\n" + "=" * 75)
print("模型比較（Precision / Recall / F1 使用 Macro 平均）")
print("=" * 75)

formatters = {
    col: "{:.4f}".format
    for col in comparison_df.select_dtypes(include="number").columns
    if col != "Feature_Count"
}
formatters["Feature_Count"] = "{:.0f}".format

from tabulate import tabulate

print(tabulate(
    comparison_df,
    headers="keys",
    tablefmt="grid",
    showindex=False,
    floatfmt=".4f"
))

# 儲存比較表為 CSV
comparison_df.to_csv(
    OUTPUT_DIR / "Model_Comparison.csv",
    index=False,
    encoding="utf-8-sig"
)
print(f"\n比較表已儲存至：{OUTPUT_DIR / 'Model_Comparison.csv'}")


# 10. 選擇最佳模型
# ============================================================

all_f1 = all_search.best_score_
top5_f1 = top5_search.best_score_

if top5_f1 >= all_f1:
    best_name = "Top5_Features"
    best_model = top5_tree_model
    best_features = top5_features
    best_result = top5_result
else:
    best_name = "All_Features"
    best_model = all_tree_model
    best_features = list(X_train.columns)
    best_result = all_result

print(f"\n依交叉驗證平均 F1 選出的模型：{best_name}")

import joblib

best_pipeline = top5_model if best_name == "Top5_Features" else all_model

joblib.dump(
    {
        "model": best_pipeline,
        "feature_columns": list(X_train.columns),
        "target": TARGET,
        "positive_class": POSITIVE_CLASS if len(class_order) == 2 else None,
    },
    OUTPUT_DIR / "best_model.joblib"
)


# ============================================================
# 11. 定義決策樹輸出函式
# ============================================================

def save_tree_outputs(model, feature_names, prefix):
    """儲存最佳模型的決策樹圖與完整文字規則。"""

    # 依實際樹大小決定畫布，避免小樹被過寬的留白拉開。
    leaf_count = model.tree_.n_leaves
    tree_depth = model.tree_.max_depth
    figure_width = max(12, leaf_count*1.5)
    figure_height = max(8, (tree_depth + 1))
    font_size = max(10, 12 - tree_depth)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=[str(label) for label in class_order],
        filled=True,
        rounded=True,
        fontsize=font_size,
        impurity=False,
        proportion=True,
        precision=2,
        ax=ax
    )

    ax.set_title(f"Decision Tree - {prefix}")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"Tree_{prefix}.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    # 儲存完整規則，不限制深度
    rules = export_text(
        model,
        feature_names=list(feature_names)
    )

    with open(
        OUTPUT_DIR / f"Rules_{prefix}.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(rules)


# ============================================================
# 12. ROC Curve：只輸出最佳模型
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    best_result["fpr"],
    best_result["tpr"],
    label=f'{best_name} ({best_result["Curve_Description"]} AUC={best_result["ROC_AUC_Curve"]:.3f})'
)

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Model")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ROC_Best_Model.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# ============================================================
# 13. PRC：只輸出最佳模型
# ============================================================

positive_rate = ((y_test == POSITIVE_CLASS).mean()
                 if len(class_order) == 2 else 1 / len(class_order))

plt.figure(figsize=(8, 6))

plt.plot(
    best_result["recall"],
    best_result["precision"],
    label=f'{best_name} ({best_result["Curve_Description"]} AP={best_result["AP_Curve"]:.3f})'
)

# 隨機分類器基準：二元分類為正類比例；多元 Micro PRC 約為 1 / 類別數
plt.axhline(
    y=positive_rate,
    linestyle="--",
    label=f"Baseline ({positive_rate:.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Best Model")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "PRC_Best_Model.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# ============================================================
# 14. 最佳模型特徵重要性排名
# 若為前五特徵模型：使用前五特徵選擇器(只用來選擇特徵)的評分(!=對Top5樹的重要性)
# ============================================================

if best_name == "Top5_Features":
    # 使用前五特徵選擇器內部決策樹的評分
    importance_values = (
        top5_model
        .named_steps["top5_selector"]
        .estimator_
        .feature_importances_
    )
    title = "Feature Importance Ranking - Top5 Selector"

else:
    # 使用全特徵最佳決策樹的評分
    importance_values = all_tree_model.feature_importances_
    title = "Feature Importance Ranking - All-Feature Model"


# 整理所有原始特徵的重要性
best_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance_values
}).sort_values(
    by="Importance",
    ascending=True
)

# 繪製完整排名
plt.figure(figsize=(10, max(6, len(best_importance_df) * 0.3)))

plt.barh(
    best_importance_df["Feature"],
    best_importance_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(title)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Feature_Importance_Best_Model.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# ============================================================
# 15. 只輸出最佳模型的決策樹與規則
# ============================================================

save_tree_outputs(
    best_model,
    best_features,
    best_name
)


# ============================================================
# 16. 結束
# ============================================================

print("\n" + "=" * 60)
print("程式執行完成")
print("=" * 60)

print(f"結果輸出位置：{OUTPUT_DIR}")
print(f"最佳模型：{best_name}")
print(f"執行時間：{time.time() - start_time:.2f} 秒")
