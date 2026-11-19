from pathlib import Path
import sys

import joblib
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEVELOPMENT_PATH = PROJECT_DIR / "data" / "processed" / "development.csv"
MODEL_PATH = PROJECT_DIR / "models" / "final_pipeline.joblib"

sys.path.insert(0, str(PROJECT_DIR / "feature_b"))
from common import load_development

st.set_page_config(page_title="Student Dropout")
st.title("Dự đoán tình trạng học tập")
st.write("Chọn hồ sơ để xem thông tin và dự đoán từ dữ liệu đến hết học kỳ 2.")

if not DEVELOPMENT_PATH.is_file():
    st.error("Không tìm thấy data/processed/development.csv.")
    st.stop()

X_a, X_ab, _ = load_development(DEVELOPMENT_PATH)
sample_index = st.selectbox(
    "Hồ sơ sinh viên",
    options=range(min(5, len(X_a))),
    format_func=lambda index: f"Hồ sơ mẫu {index + 1}",
)

student = X_a.iloc[sample_index]
registered, passed, grade = st.columns(3)
registered.metric("Môn đăng ký HK1", int(student["Curricular units 1st sem (enrolled)"]))
passed.metric("Môn đạt HK1", int(student["Curricular units 1st sem (approved)"]))
grade.metric("Điểm HK1", f'{student["Curricular units 1st sem (grade)"]:.2f}')

if st.button("Dự đoán", type="primary"):
    if not MODEL_PATH.is_file():
        st.warning("Hãy chạy notebook 06 để tạo models/final_pipeline.joblib.")
        st.stop()

    saved = joblib.load(MODEL_PATH)
    features = {"A": X_a, "A+B": X_ab}[saved["feature_set"]]
    prediction = saved["pipeline"].predict(features.iloc[[sample_index]])[0]
    label = saved["labels"][int(prediction)]
    translations = {"Dropout": "Bỏ học", "Enrolled": "Đang học", "Graduate": "Tốt nghiệp"}

    st.success(f"Dự đoán: {translations[label]} ({label})")
    st.caption(f"Mô hình: {saved['model_name']} · Bộ đặc trưng: {saved['feature_set']}")