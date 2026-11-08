# ML01 — Dự đoán trạng thái học tập của sinh viên

Báo cáo tiến độ giữa kỳ môn Máy học · Nhóm 08 · Lớp DCT124C3

Đề tài dự đoán một trong ba trạng thái của sinh viên gồm Dropout (bỏ học), Enrolled (đang học) và Graduate (tốt nghiệp). Bộ dữ liệu có 4.424 mẫu và 36 đặc trưng đầu vào. Nguồn dữ liệu là [Predict Students’ Dropout and Academic Success — UCI Machine Learning Repository](https://doi.org/10.24432/C5MC89).

## Thành viên và phân công thực nghiệm

| Thành viên | MSSV | Phụ trách |
|---|---|---|
| Lê Đỗ Hòa Thương | 3124411304 | Random Forest |
| Nguyễn Hoàng Khởi | 3124411137 | XGBoost |
| Huỳnh Lê Khánh Ly | 3124411160 | Logistic Regression |
| Trần Minh Triết (nhóm trưởng) | 3124411321 | SVM (RBF) |

## Cấu trúc dự án

| Đường dẫn | Nội dung |
|---|---|
| `data/raw/student_dropout.csv` | Dữ liệu gốc từ UCI |
| `data/processed/` | `development.csv` (3.539 mẫu) và `test.csv` (885 mẫu) |
| `notebooks/` | Notebook EDA và bốn notebook thực nghiệm |
| `src/evaluate.py` | Tiện ích Python liên quan đến đánh giá và nhật ký |
| `experiments/` | Nhật ký Excel, CSV kết quả tìm kiếm và báo cáo OOF theo thành viên |
| `report/figures/` | Biểu đồ EDA và biểu đồ kết quả thực nghiệm |
| `report/tables/` | Bảng thống kê xuất từ notebook EDA |
| `models/` | Pipeline ứng viên `candidate_pipeline.joblib` và `metadata.json` trong các thư mục `thuong/`, `khoi/`, `ly/`, `triet/` |
| `requirements.txt`, `requirements_training.txt` | Danh sách thư viện và khoảng phiên bản dùng khi chạy notebook |

## Cài đặt và chạy lại

Mở thư mục `ML01_Student_Dropout` trong VS Code. Tại PowerShell trong thư mục này, tạo môi trường Python và cài các thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements_training.txt
```

Trong VS Code, mở notebook và chọn kernel Python thuộc môi trường `.venv`. Chạy các ô theo thứ tự từ trên xuống.

1. Chạy `notebooks/01_eda.ipynb` để khảo sát dữ liệu và tạo hai tập CSV trong `data/processed/`.
2. Sau khi có `development.csv`, chạy các notebook mô hình. Các notebook này dùng chung dữ liệu phát triển và có thể chạy độc lập với nhau.
   - `notebooks/02_random_forest.ipynb`
   - `notebooks/03_xgboost.ipynb`
   - `notebooks/04_logistic_regression.ipynb`
   - `notebooks/05_svm.ipynb`
3. Xem kết quả và nhật ký trong `experiments/`, biểu đồ trong `report/figures/` và cấu hình ứng viên trong `models/`. Nếu cần đối chiếu môi trường dùng để lưu một pipeline, xem phiên bản scikit-learn trong `metadata.json` của mô hình tương ứng.

Ô vẽ biểu đồ GridSearchCV trong notebook Random Forest đọc CSV của lượt thực nghiệm dùng trong báo cáo. Giữ file `experiments/thuong/20261031_144833_3c4529_grid_search.csv` để chạy ô này. Khi muốn vẽ kết quả của lượt mới, cập nhật `csv_path` trong ô vẽ sang file `*_grid_search.csv` vừa sinh ra.

## Thiết kế và kết quả giữa kỳ

Nhóm chia dữ liệu theo tỷ lệ 80/20 có phân tầng theo nhãn, với `random_state=42`. Bốn mô hình dùng cùng tập phát triển và đánh giá chéo phân tầng 5 fold với seed 42. Macro-F1 là chỉ số chính để chọn siêu tham số. Nhóm sử dụng dự đoán ngoài fold (OOF) để phân tích lỗi theo từng lớp.

| Mô hình | CV Macro-F1 ban đầu | CV Macro-F1 sau tinh chỉnh |
|---|---:|---:|
| Dummy (mốc đối chứng) | 0,2220 | — |
| Logistic Regression | 0,7013 | 0,7174 |
| Random Forest | 0,6766 | 0,7162 |
| XGBoost | 0,7063 | 0,7136 |
| SVM (RBF) | 0,6858 | 0,7154 |

Logistic Regression hiện đạt CV Macro-F1 cao nhất trong các cấu hình đã thử. Kết quả giữa kỳ sử dụng 36 đặc trưng gốc. Tỷ lệ học phần đạt trong học kỳ 1 mới được khảo sát ở bước EDA và sẽ được đánh giá trong thực nghiệm bổ sung. Tập test 885 mẫu được giữ riêng để đánh giá sau khi chốt mô hình.
