"""So sánh A với A+B bằng cấu hình đã chọn trên A và cùng 5 fold."""

from pathlib import Path

import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LABELS = ["Dropout", "Enrolled", "Graduate"]
CATEGORICAL_COLUMNS = ['Marital status', 'Application mode', 'Course', 'Previous qualification',
    'Nacionality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
BINARY_COLUMNS = ['Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
NUMERIC_COLUMNS = ['Application order', 'Previous qualification (grade)', 'Admission grade',
    'Age at enrollment', 'Unemployment rate', 'Inflation rate', 'GDP'] + [
    f'Curricular units {semester} sem ({measure})'
    for semester in ['1st', '2nd'] for measure in [
        'credited', 'enrolled', 'evaluations', 'approved', 'grade', 'without evaluations']]

# Chuẩn bị dữ liệu a và a+b
# Đọc development.csv, kiểm tra dữ liệu và tách cột Target thành nhãn y.
# A giữ 36 đặc trưng gốc. A+B thêm tỷ lệ đạt HK1 và cờ đăng ký 0 môn.
# Trả về dữ liệu A, dữ liệu A+B và nhãn y.
def load_development(path):
    data = pd.read_csv(path)
    data.columns = [" ".join(name.split()) for name in data.columns]
    expected = set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + BINARY_COLUMNS + ["Target"])
    if data.columns.has_duplicates or set(data.columns) != expected:
        raise ValueError("development.csv phải có đúng 36 đặc trưng gốc và Target.")
    if data.isna().any().any() or set(data["Target"]) != set(LABELS):
        raise ValueError("Kiểm tra giá trị thiếu và ba nhãn trong development.csv.")

    y = data["Target"].map({name: index for index, name in enumerate(LABELS)})
    features_a = data.drop(columns="Target")
    enrolled = features_a["Curricular units 1st sem (enrolled)"]
    approved = features_a["Curricular units 1st sem (approved)"]
    zero_enrolled = enrolled.eq(0)
    pass_rate = approved.div(enrolled.mask(zero_enrolled)).mask(zero_enrolled, 0)

    features_ab = features_a.assign(
        hk1_pass_rate=pass_rate,
        hk1_zero_enrolled=zero_enrolled.astype(int),
    )
    print(f"{len(y)} mẫu | A: {features_a.shape[1]} cột | A+B: {features_ab.shape[1]} cột")
    print(f"Số mẫu đăng ký 0 môn HK1: {zero_enrolled.sum()}")
    return features_a, features_ab, y

# ghép tiền xử lý với mô hình
# Mã hóa One-Hot các cột danh mục và giữ nguyên các cột nhị phân.
# Chuẩn hóa cột số nếu scale_numeric=True, như với Logistic Regression và SVM.
# Nếu add_b=True, đưa hai cột B vào đúng nhóm rồi ghép tiền xử lý với mô hình.
def make_pipeline(model, scale_numeric, add_b):
    numeric = NUMERIC_COLUMNS + (["hk1_pass_rate"] if add_b else [])
    binary = BINARY_COLUMNS + (["hk1_zero_enrolled"] if add_b else [])
    preprocessing = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("numeric", StandardScaler() if scale_numeric else "passthrough", numeric),
        ("binary", "passthrough", binary),
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", clone(model))])

# Đánh giá một phiên bản dữ liệu (A hoặc A+B) qua 5 fold.
# Mỗi lượt học trên 4 fold, dự đoán trên fold còn lại và tính Macro-F1.
# Lấy trung bình Macro-F1 của 5 lượt và tính Recall từng lớp từ dự đoán OOF.
def evaluate(name, model, scale_numeric, version, features, y, folds):
    oof_predictions = pd.Series(index=y.index, dtype="int64")
    fold_results = []
    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        pipeline = make_pipeline(model, scale_numeric, version == "A+B")
        pipeline.fit(features.iloc[train_idx], y.iloc[train_idx])
        predictions = pipeline.predict(features.iloc[valid_idx])
        oof_predictions.iloc[valid_idx] = predictions
        fold_results.append({
            "model": name,
            "data": version,
            "fold": fold_number,
            "macro_f1": f1_score(y.iloc[valid_idx], predictions, average="macro"),
        })

    scores = pd.DataFrame(fold_results)
    recalls = recall_score(y, oof_predictions.astype(int), labels=[0, 1, 2], average=None)
    summary = {
        "model": name,
        "data": version,
        "features": features.shape[1],
        "cv_macro_f1": scores["macro_f1"].mean(),
        **{f"oof_recall_{label.lower()}": score for label, score in zip(LABELS, recalls)},
    }
    return summary, scores

# Chạy và so sánh 2 phiên bản
# Tạo một bộ 5 fold dùng chung cho cả A và A+B.
# Gọi evaluate hai lần với cùng cấu hình mô hình, mỗi lần cho một phiên bản.
# In mức thay đổi Macro-F1 và lưu bảng tổng hợp cùng điểm từng fold ra hai CSV.
def run_comparison(model_name, model, *, scale_numeric):
    project_dir = Path(__file__).resolve().parents[1]
    features_a, features_ab, y = load_development(
        project_dir / "data" / "processed" / "development.csv"
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(features_a, y))

    rows = []
    fold_tables = []
    for version, features in [("A", features_a), ("A+B", features_ab)]:
        summary, scores = evaluate(model_name, model, scale_numeric, version, features, y, folds)
        rows.append(summary)
        fold_tables.append(scores)

    results = pd.DataFrame(rows)
    results["sklearn_version"] = sklearn.__version__
    output_dir = project_dir / "experiments" / "feature_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{model_name}_a_vs_ab.csv"
    results.to_csv(summary_path, index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(
        output_dir / f"{model_name}_a_vs_ab_folds.csv", index=False
    )

    print(results.round(6).to_string(index=False))
    delta = rows[1]["cv_macro_f1"] - rows[0]["cv_macro_f1"]
    print(f"Chênh lệch CV Macro-F1 (A+B trừ A): {delta:+.6f}")
    print(f"Đã lưu: {summary_path}")
