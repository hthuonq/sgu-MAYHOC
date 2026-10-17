import os
from datetime import datetime
import openpyxl
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def log_experiment(
    model_name, hyperparameters, metrics_dict, run_by="Thương"
):
    """Tự động tạo/cập nhật file Excel experiments/experiment_history.xlsx"""
    excel_path = os.path.join("experiments", "experiment_history.xlsx")
    os.makedirs("experiments", exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_data = {
        "Timestamp": [current_time],
        "Model_Name": [model_name],
        "Hyperparameters": [str(hyperparameters)],
        "Accuracy": [round(metrics_dict.get("accuracy", 0), 4)],
        "Macro_F1": [round(metrics_dict.get("macro_f1", 0), 4)],
        "Weighted_F1": [round(metrics_dict.get("weighted_f1", 0), 4)],
        "Run_By": [run_by],
    }

    df_new = pd.DataFrame(new_data)

    if not os.path.exists(excel_path):
        df_new.to_excel(excel_path, index=False, engine="openpyxl")
        print(f" Đã tạo mới file Excel log tại: {excel_path}")
    else:
        with pd.ExcelWriter(
            excel_path, mode="a", engine="openpyxl", if_sheet_exists="overlay"
        ) as writer:
            sheet = writer.book.active
            start_row = sheet.max_row
            df_new.to_excel(
                writer, index=False, header=False, startrow=start_row
            )
        print(f" Đã lưu kết quả [{model_name}] của {run_by} vào Excel log!")


def evaluate_and_log(
    model, X_test, y_test, model_name, hyperparameters, run_by="Thương"
):
    """Hàm chạy dự đoán và tự động đẩy số liệu ra Excel log"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    metrics = {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

    print(f"\n--- KẾT QUẢ MÔ HÌNH: {model_name} (Thực hiện: {run_by}) ---")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred))

    log_experiment(model_name, hyperparameters, metrics, run_by=run_by)
    return metrics