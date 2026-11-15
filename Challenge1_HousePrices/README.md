# Challenge 1 — House Prices

Dự án môn Máy học tại Trường Đại học Sài Gòn. Nhóm dự đoán giá bán nhà (`SalePrice`) từ bộ dữ liệu Ames Housing và so sánh Ridge, Random Forest, XGBoost, CatBoost trên cùng cách chia kiểm định chéo 5 fold.

Thành viên: Nguyễn Hoàng Khởi, Huỳnh Lê Khánh Ly, Lê Đỗ Hoà Thương, Trần Minh Triết.

## Cài đặt trên Windows

Cần Git, Python 3.11 hoặc 3.12 và VS Code có hai extension **Python** và **Jupyter**. Mở PowerShell rồi chạy:

```powershell
git clone https://github.com/hthuonq/sgu-MAYHOC.git
cd sgu-MAYHOC/Challenge1_HousePrices
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
code .
```

Nếu đã clone repo, mở PowerShell ở thư mục `sgu-MAYHOC` và bắt đầu từ `cd Challenge1_HousePrices`. Nếu dùng Python 3.11, đổi `py -3.12` thành `py -3.11`. Nếu lệnh `code .` không có sẵn, mở thư mục `Challenge1_HousePrices` từ **File → Open Folder** trong VS Code.

Dữ liệu có sẵn trong [`data/raw/`](./data/raw/), gồm `train.csv`, `test.csv`, `sample_submission.csv` và mô tả các cột. Nguồn: [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Chạy notebook

Mở từng notebook, bấm **Select Kernel → Python Environments**, chọn Python trong `.venv` của `Challenge1_HousePrices`, rồi bấm **Run All**.

| Thứ tự | Notebook | Kết quả |
| --- | --- | --- |
| 1 | [Khảo sát dữ liệu](./notebooks/01_eda.ipynb) | Biểu đồ phân phối, giá trị thiếu và quan hệ giữa đặc trưng với giá bán |
| 2 | [Ridge](./notebooks/02_ridge.ipynb) | Điểm kiểm định chéo và CSV dự đoán của Ridge |
| 3 | [Random Forest](./notebooks/03_random_forest.ipynb) | Điểm kiểm định chéo và CSV dự đoán của Random Forest |
| 4 | [XGBoost](./notebooks/04_xgboost.ipynb) | Điểm kiểm định chéo và CSV dự đoán của XGBoost |
| 5 | [CatBoost](./notebooks/05_catboost.ipynb) | Điểm kiểm định chéo và CSV dự đoán của CatBoost |
| 6 | [So sánh bốn mô hình](./notebooks/comparate_models.ipynb) | Tổng hợp kết quả CV, vẽ biểu đồ và tạo tệp dự đoán cuối từ kết quả của notebook 02–05 |

Các notebook mô hình chạy độc lập và dùng các hàm chung trong [`src/`](./src/). Nếu gặp `No module named src`, mở lại đúng thư mục `Challenge1_HousePrices`. Nếu thiếu thư viện như `catboost`, kiểm tra kernel đang trỏ đến `.venv` đã cài các gói từ `requirements.txt`.

Sau khi chạy các mô hình, kiểm tra [`results/cv/`](./results/cv/) và [`results/submissions/`](./results/submissions/). Mỗi notebook cập nhật CSV của mô hình tương ứng. Bảng tổng hợp, biểu đồ so sánh và tệp nộp cuối bên dưới là các bản đã lưu trong repo.

## Kết quả thực nghiệm

Mỗi mô hình được đánh giá với bộ cơ bản gồm 79 đặc trưng và bộ bổ sung gồm 82 đặc trưng trước khi tinh chỉnh. Bảng ghi RMSE trung bình trên log giá bán qua 5 fold của cấu hình được chọn. RMSE càng thấp càng tốt.

| Mô hình | Bộ đặc trưng | Tham số được chọn | CV RMSE |
| --- | --- | --- | ---: |
| CatBoost | Bổ sung | `depth = 5` | **0,12388** |
| XGBoost | Bổ sung | `max_depth = 3` | 0,13211 |
| Random Forest | Bổ sung | `min_samples_leaf = 3` | 0,14303 |
| Ridge | Cơ bản | `alpha = 30` | 0,14657 |

- [Bảng kết quả đầy đủ](./results/results_summary.csv) và [biểu đồ so sánh](./results/figures/compare_models.png)
- [Tệp dự đoán CatBoost đã nộp](./results/submissions/submission_final.csv) và [minh chứng điểm Kaggle 0,12680](./minhchung/kaggle_score_catboost.jpg)

## Báo cáo và trình bày

- [Paper](./report/paper.docx)
- [Slide trình bày](./report/Challenge1_HousePrices.pptx)
- [Poster](./report/Poster_House_Prices.pptx)
