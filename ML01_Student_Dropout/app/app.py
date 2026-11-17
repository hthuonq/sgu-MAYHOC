from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEVELOPMENT_PATH = PROJECT_DIR / "data" / "processed" / "development.csv"

st.set_page_config(page_title="Student Dropout")
st.title("Dự đoán tình trạng học tập")
st.write("Chọn một hồ sơ mẫu để xem thông tin học kỳ 1.")

if not DEVELOPMENT_PATH.is_file():
    st.error("Không tìm thấy data/processed/development.csv.")
    st.stop()

samples = pd.read_csv(DEVELOPMENT_PATH, nrows=5)
samples.columns = [" ".join(column.split()) for column in samples.columns]

sample_index = st.selectbox(
    "Hồ sơ sinh viên",
    options=range(len(samples)),
    format_func=lambda index: f"Hồ sơ mẫu {index + 1}",
)

student = samples.iloc[sample_index]

registered, passed, grade = st.columns(3)
registered.metric("Môn đăng ký HK1", int(student["Curricular units 1st sem (enrolled)"]))
passed.metric("Môn đạt HK1", int(student["Curricular units 1st sem (approved)"]))
grade.metric("Điểm HK1", f'{student["Curricular units 1st sem (grade)"]:.2f}')

if st.button("Dự đoán", type="primary"):
    st.info("Chưa có mô hình cuối. Nhóm sẽ gắn pipeline sau khi chốt thực nghiệm.")