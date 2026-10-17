# Đề tài ML01: Dự đoán Sinh viên Dropout / Enrolled / Graduate

## Danh sách thành viên Nhóm 08 (Lớp DCT124C3):
- **Lê Đỗ Hòa Thương** - MSSV: 3124411304
- **Nguyễn Hoàng Khởi** - MSSV: 3124411137
- **Huỳnh Lê Khánh Ly** - MSSV: 3124411160
- **Trần Minh Triết (Trưởng nhóm)** - MSSV: 3124411321

---

## Cấu trúc thư mục dự án
- `data/`: Chứa dữ liệu gốc (`raw/student_dropout.csv`) và dữ liệu đã qua xử lý (`processed/`).
- `notebooks/`: Chứa các Jupyter Notebook thực nghiệm (`01_eda.ipynb`, `02_model_training.ipynb`, `03_evaluation.ipynb`).
- `src/`: Mã nguồn Python tái sử dụng (`data_loader.py`, `evaluate.py`).
- `models/`: Lưu trữ các mô hình đã huấn luyện (`best_model.pkl`).
- `app/`: Mã nguồn Web Demo Streamlit (`app.py`).
- `experiments/`: Tự động lưu lịch sử thực nghiệm ra file Excel (`experiment_history.xlsx`).
- `report/`: Báo cáo Word, Slide thuyết trình và Poster đề tài.

---

## Hướng dẫn cài đặt cho thành viên
Mở Terminal tại thư mục dự án và chạy lệnh cài đặt thư viện:
```bash
pip install -r requirements.txt