from xgboost import XGBClassifier

from common import run_comparison


if __name__ == "__main__":
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
    )
    run_comparison("xgb", model, scale_numeric=False)
