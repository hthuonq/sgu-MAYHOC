from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
SEED = 42


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    target = train.pop("SalePrice")
    train.pop("Id")
    ids = test.pop("Id")
    if (target <= 0).any() or not np.isfinite(target).all():
        raise ValueError("SalePrice phải hữu hạn và lớn hơn 0.")
    if set(train.columns) != set(test.columns):
        raise ValueError("Các cột đặc trưng của train và test phải giống nhau.")
    if not ids.is_unique or not sample["Id"].is_unique or set(ids) != set(sample["Id"]):
        raise ValueError("Id của test phải khớp file mẫu và không được trùng.")
    return train, np.log(target), test[train.columns], ids, sample


def make_folds():
    return KFold(n_splits=5, shuffle=True, random_state=SEED)


def save_submission(name, predicted_log, ids, sample):
    prices = np.exp(predicted_log)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Giá dự đoán phải hữu hạn và lớn hơn 0.")
    submission = sample[["Id"]].assign(SalePrice=sample["Id"].map(pd.Series(prices, index=ids)))
    path = RESULTS / "submissions" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission
