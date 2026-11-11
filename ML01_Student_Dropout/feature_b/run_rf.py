
from sklearn.ensemble import RandomForestClassifier

from common import run_comparison


if __name__ == "__main__":
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    run_comparison("rf", model, scale_numeric=False)
